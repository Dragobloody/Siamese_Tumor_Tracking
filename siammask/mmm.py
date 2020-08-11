import sys
sys.path.append('D:/MasterAIThesis/code/load_dicom/')
import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

import index_tracker as track   
import csv
from keras.callbacks import LearningRateScheduler
import cv2
import scipy
import h5py
import skimage
import time

# set seed
from numpy.random import seed
seed(1234)
from tensorflow import set_random_seed
set_random_seed(1234)

from siammask import utils, resnet, rpn as rp, proposal, detection, fpn, losses, data_generator as dg, visualize, metrics
from siammask.config import Config


#from MaskRCNN import utils, resnet, rpn as rp, proposal, detection, fpn, losses, data_generator as dg, visualize
#from MaskRCNN.config import Config

from siammask import siammask_model as modellib
#from MaskRCNN import maskrcnn_model as modellib
import dicom_to_numpy as dtn
import matplotlib.pyplot as plt

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


def fit_mask(mask, pos, image_shape = (255, 255)):
    mask_shape = mask.shape
    final_mask = np.zeros(image_shape, dtype = mask.dtype)
    shift_x = mask_shape[0]//2+mask_shape[0]%2
    shift_y = mask_shape[1]//2+mask_shape[1]%2
    x_center = shift_x + pos[0]*8
    y_center = shift_y + pos[1]*8
    final_mask[x_center-mask_shape[0]//2:x_center+shift_x,
               y_center-mask_shape[1]//2:y_center+shift_y] = mask
    
    return final_mask


# PRINT SOME PREDICTIONS 
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# fit an ellipse to segmentation mask
def fit_ellipse(mask):
    mask = scipy.ndimage.binary_fill_holes(mask).astype('uint8')
    non_zero_points = cv2.findNonZero(mask)
    elps = cv2.fitEllipse(non_zero_points)
    result = cv2.ellipse(np.zeros(mask.shape, 'uint8'), elps, (1, 1, 1), -1)
    
    return result



class TumorConfig(Config):    
    NAME = 'lung_tumor'
    BACKBONE = "resnet50"
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,   
        "rpn_centerness_loss":1.,        
        "rpn_mask_loss": 1.        
    }
    LEARNING_RATE = 0.001
    IMAGES_PER_GPU = 16
    TRAIN_BN = True                   
    IMAGE_HEIGHT = 384
    IMAGE_WIDTH = 512
    IMAGE_SHAPE = [384, 512, 1]
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 181
    
    BACKBONE_STRIDES = [8]
    RPN_ANCHOR_SCALES = [64]
    
    POST_NMS_ROIS_TRAINING = 1024
    POST_NMS_ROIS_INFERENCE = 512
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    RPN_NMS_THRESHOLD = 0.8
    BBOX_SIGMA = 2.0
    BBOX_LOSS_LAMDA_D = 10
    BBOX_LOSS_LAMDA_C = 1
    
    TRAIN_MASKS_PER_IMAGE = 20
    
    ANCHORLESS = True 
    SIAMRPN = False 
    CROP = False
    
    MASK = False
    REFINE = False    
    STAGE4 = True
    
    BBOX_RATIO = 0.8
    
    KERNEL_IMAGE_SIZE = 128
    IMAGE_CHANNEL_COUNT = 3
    if CROP:
        FEATURES_SHAPE = [25, 25]
    else:
        FEATURES_SHAPE = [17, 17]
    


# SET UP CONFIG
config = TumorConfig()

data_path = 'D:/MasterAIThesis/h5py_data/vumc patient data/'
patient_name = 'patient2'
model = 'siammask'
data_mode = 'decorrelated'
drr_data_path = data_path + patient_name + '/models/'+ model + '/' + data_mode +'/dataset/'
# Directory to save logs and trained model
LOG_DIR =  data_path + patient_name + '/models/' + model + '/' + data_mode +'/logs/'

test_imgs = dtn.load_data(drr_data_path + 'test_imgs.hdf5', 'test_imgs')
test_labs = dtn.load_data(drr_data_path + 'test_labs.hdf5', 'test_labs')
for i in range(test_labs.shape[0]):
    for j in range(test_labs.shape[1]):
        test_labs[i, j] = fit_ellipse(test_labs[i, j])
        
test_imgs = (test_imgs+1)/2.



model = modellib.SiamMask(mode="inference", config=config,
                                      model_dir=LOG_DIR)

model.keras_model.summary()

weights_path = model.find_last()
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True) 



# -------------------------------------------------------------------------


def init_bbox(image_shape, bbox_size = 64):
    shape = np.array(search_image.shape)
    search_center = shape//2
    h = w = bbox_size//2
    kernel_bbox = [search_center[0]-w, search_center[1]-h,
                       search_center[0]+w, search_center[1]+h]
    kernel_bbox = np.array(kernel_bbox, dtype='int32')[np.newaxis, ...]
    
    return kernel_bbox
           

def get_max_results(results):
    max_score = 0
    max_idx = 
    for i, r in enumerate(results):
        idx = np.argmax(r['scores'])
        if r['scores'][idx] > max_score:
            max_score = r['scores'][idx]
            max_bbox = r['rois'][idx]
            max_mask = r['masks'][idx]
    
    return max_score, max_bbox, max_mask



def nearest_neighbor(z_i, z_t, z):
    nn = np.linalg.norm(z_i-z_t)
    for z_j in z:
        if np.linalg.norm(z_i-z_j) < nn:
            return 0
    return 1
        
def reverse_nearest_neighbor(memory, z_t):
    z = memory['z']
    for i in range(len(z)):
        if nearest_neighbor(z[i], z_t, z[:i]+z[i+1:])==1:
            return z    
    memory['z'].append(z_t)
    memory['z_w'].append(1.0)
    
    for i in range(len(memory['z'])):
        if memory['z_w'][i] < 0:
            memory['z'].pop(i)
            memory['z_w'].pop(i)
    
    return memory

z_prime = reverse_nearest_neighbor(z, z_t)


def tracking(model_local, model_global, search_image, memory, t_1, t_2, t_3):
    if len(memory['states']) == 0 or memory['states'][-1]=='LOST':
        kernel_bbox = init_bbox(search_image.shape, bbox_size = 64)
        
        z, s, \
        s_meta, _, \
        _, _, _, _, _ = dg.load_image_gt(model_global.config, 
                                search_image, search_image, 
                                kernel_mask=None, search_mask=None, 
                                kernel_bbox=kernel_bbox,
                                mode = model_global.mode)   

        results = model_global.detect(z[np.newaxis, ...], 
                                     s[np.newaxis, ...], 
                                     s_meta[np.newaxis, ...]) 
        
        max_score, max_bbox, max_mask = 
                        
    
    if memory['states'][-1]=='STABLE':
        max_score = 0
        max_bbox = memory['bboxs'][-1]
        max_mask = memory['masks'][-1]
        max_idx = len(memory['z'])-1
        
        for i in range(len(memory['z'])):
            score, bbox, mask = model_local.predict(memory['z'][i], search_image)
            if score >= max_score:
                max_score = score
                max_bbox = bbox
                max_mask = mask
                max_idx = i
        
        memory['scores'].append(max_score)
        memory['bboxs'].append(max_bbox)
        memory['masks'].append(max_mask)
        
        for j in range(len(memory['z_w'])):
            if j == max_idx:
                memory['z_w'][j] += 0.4
            else:
                memory['z_w'][j] -= 0.2               
        
        if np.mean(memory['scores'][-3:]) < t_2:
            memory['states'].append('LOST')
        else:
            memory['states'].append('STABLE')
        
                
def init_memory():
    memory = {}                
    memory['states'] = []               
    memory['z'] = []     
    memory['z_w'] = []           
    memory['scores'] = [] 
    memory['bboxs'] = [] 
    memory['masks'] = [] 

    return memory              
                
                
                
memory = init_memory()             
model_local = model_global = model             
                
                
search_image = test_imgs[0, 0]                
                
                
                
                
                



        
        
        