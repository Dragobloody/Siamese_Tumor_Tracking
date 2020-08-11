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

from semseg import utils, resnet, detection, losses, data_generator as dg, visualize, metrics
from semseg.config import Config


#from MaskRCNN import utils, resnet, rpn as rp, proposal, detection, fpn, losses, data_generator as dg, visualize
#from MaskRCNN.config import Config

from semseg import semseg_model as modellib
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


"""

class TumorConfig(Config):    
    NAME = 'lung_tumor'
    LOSS_WEIGHTS = {           
        "rpn_mask_loss": 1.        
    }
    LEARNING_RATE = 0.001
    IMAGES_PER_GPU = 1
    TRAIN_BN = True  
    IMAGE_HEIGHT = 384
    IMAGE_WIDTH = 384
    IMAGE_SHAPE = [384, 384, 1]
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 181   
    
    IMAGE_CHANNEL_COUNT = 1
   

# SET UP CONFIG
config = TumorConfig()

data_path = 'D:/MasterAIThesis/h5py_data/vumc patient data/'
patient_name = 'patient1'
model = 'semseg'
data_mode = 'standard'
drr_data_path = data_path + patient_name + '/models/'+ model + '/' + data_mode +'/dataset/'
# Directory to save logs and trained model
LOG_DIR =  data_path + patient_name + '/models/' + model + '/' + data_mode +'/logs/'


imgs = dtn.load_data(drr_data_path + 'train_imgs.hdf5', 'train_imgs')
labs = dtn.load_data(drr_data_path + 'train_labs.hdf5', 'train_labs')
test_imgs = dtn.load_data(drr_data_path + 'test_imgs.hdf5', 'test_imgs')
test_labs = dtn.load_data(drr_data_path + 'test_labs.hdf5', 'test_labs')
for i in range(labs.shape[0]):
    for j in range(labs.shape[1]):
        labs[i, j] = fit_ellipse(labs[i, j])
for i in range(test_labs.shape[0]):
    for j in range(test_labs.shape[1]):
        test_labs[i, j] = fit_ellipse(test_labs[i, j])
        
        
imgs = (imgs+1)/2.
test_imgs = (test_imgs+1)/2.

# TRAIN MODEL
s = test_imgs.shape
imgs = np.reshape(imgs, (-1, s[2], s[3]))[:, :, 64:-64]
labs = np.reshape(labs, (-1, s[2], s[3]))[:, :, 64:-64]
test_imgs = np.reshape(test_imgs, (-1, s[2], s[3]))[:, :, 64:-64]
test_labs = np.reshape(test_labs, (-1, s[2], s[3]))[:, :, 64:-64]

    

# clear memory 1
K.clear_session()   


model = modellib.SiamMask(mode="training", config=config,
                                      model_dir=LOG_DIR)
resnet_weights_path = 'D:/MasterAIThesis/code/siammask/pretrained_weights/resnet50_pretrained_on_siamese.h5'
vgg16_weights_path = 'D:/MasterAIThesis/code/siammask/pretrained_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

model.load_weights(resnet_weights_path, by_name=True, exclude='4+') 

model = modellib.SiamMask(mode="inference", config=config,
                                      model_dir=LOG_DIR)

model.keras_model.summary()

weights_path = model.find_last()
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True) 


def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch<=10 and lr < 1e-2:
        lr = lr + lr/epoch
    elif epoch > 10 and lr > 1e-5:
        k = 0.05
        lr = lr*np.exp(-k*epoch)
    return lr

lrs = LearningRateScheduler(lr_scheduler)  

model.train(imgs, labs, test_imgs, test_labs,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='all',
                custom_callbacks = [lrs])






scores = outputs[11][..., -1:]
centerness=outputs[14][..., -1:]
masks = outputs[1][...,-1]
masks_thresh = np.round(outputs[1][...,-1]).astype('uint8')
masks_ceil = (outputs[1][...,-1] > 0.1).astype('uint8')

properties = skimage.measure.regionprops(masks_thresh[9], masks[9])
w_center = properties[0].weighted_centroid
center = properties[0].centroid
bbox = properties[0].bbox
properties_1 = skimage.measure.regionprops(masks_ceil[0], masks[0])
score = properties_1[0].mean_intensity

data_generator_drrs = modellib.data_generator(test_imgs, test_labs, config, shuffle=True, batch_size=config.BATCH_SIZE, mode = model.mode)
data_generator_xrays = modellib.data_generator(xrays[np.newaxis,...], test_labs[:1, :222], shuffle=True, batch_size=config.BATCH_SIZE, mode = model.mode)

inputs_drrs, _ = next(data_generator_drrs)
inputs_xrays, _ = next(data_generator_xrays)

start = time.time()
inputs_drrs, _ = next(data_generator_drrs)
outputs = model.keras_model.predict(inputs_drrs[0])
stop = time.time()
print(stop - start)


outputs = model.keras_model.predict(inputs_xrays[0])
outputs = model.detect(inputs_drrs[0])


start = time.time()        
outputs = model.test(test_imgs[0:], labs = test_labs[0:], verbose = 0)
stop = time.time()
print(stop-start)
gt_bboxs = outputs[0]['gt_bboxs']
pred_bboxs = outputs[1]['pred_bboxs'][:, np.newaxis, :]
gt_masks = outputs[0]['gt_masks']
pred_masks = outputs[1]['pred_masks']
pred_centroids = outputs[1]['pred_centroid']



cor_z, cor_x = metrics.concordance_correlation_coefficient(gt_bboxs, pred_centroids)
z_mad, z_std, x_mad, x_std = metrics.mad(gt_bboxs, pred_centroids)    
z_mad, z_std, x_mad, x_std = z_mad/3, z_std/3, x_mad/3, x_std/3 


test_data_mode = 'decorrelated'
test_model = 'siammask'
test_data_path = data_path + patient_name + '/models/'+ test_model + '/' + test_data_mode +'/dataset/'
test_imgs = dtn.load_data(test_data_path + 'test_imgs_gans.hdf5', 'test_imgs_gans')
test_labs = dtn.load_data(test_data_path + 'test_labs.hdf5', 'test_labs')
for i in range(test_labs.shape[0]):
    for j in range(test_labs.shape[1]):
        test_labs[i, j] = fit_ellipse(test_labs[i, j])
        




# simulate test data
gt_bboxs, pred_bboxs = [], []      
gt_search_bboxs, pred_search_bboxs = [], []        
 
for i in range(test_imgs.shape[0]):
    gt, pred = model.test(test_imgs[i], test_labs[i], verbose = 0)
    gt_bboxs.append(gt['gt_bboxs'])
    pred_bboxs.append(pred['pred_bboxs'][:, np.newaxis, :])
    gt_search_bboxs.append(gt['gt_search_bboxs'])
    pred_search_bboxs.append(pred['pred_search_bboxs'][:, np.newaxis, :])
gt_bboxs = np.stack(gt_bboxs, axis = 0)
pred_bboxs = np.stack(pred_bboxs, axis = 0)
gt_search_bboxs = np.stack(gt_search_bboxs, axis = 0)
pred_search_bboxs = np.stack(pred_search_bboxs, axis = 0)

gt_bboxs = np.reshape(gt_bboxs, (-1, 1, 4))
pred_bboxs = np.reshape(pred_bboxs, (-1, 1, 4)) 
gt_search_bboxs = np.reshape(gt_search_bboxs, (-1, 1, 4))
pred_search_bboxs = np.reshape(pred_search_bboxs, (-1, 1, 4))
       



gt_masks = utils.bboxs_to_masks(gt_bboxs)
pred_masks = utils.bboxs_to_masks(pred_bboxs)
gt_masks = gt_masks[:, 0, :, :]
pred_masks = pred_masks[:, 0, :, :]

X = outputs[4]
Y = Z = np.zeros(X.shape, dtype='uint8')

X = np.transpose(image, (1, 2, 0))[np.newaxis,...]
Z = Y = np.zeros(X.shape, dtype='float32')
Z = Y = X
X = np.transpose(imgs, (0, 2, 3, 1))
Z = Y = np.transpose(labs, (0, 2, 3, 1))

Z = Y = np.zeros(X.shape, dtype='float32')

#X = imgs[5][np.newaxis, ...]
Z = Y = np.transpose(test_labs, (0, 2, 3, 1))
#Z = new_lab[np.newaxis, ...]
Z= np.transpose(non_zero_ellipse_labs, (0, 2, 3, 1))
#Z = np.transpose(pred_masks, (0, 2, 3, 1))
#Z = np.transpose(pred['pred_masks'],(1, 2, 0))[np.newaxis, ...]

X = np.transpose(test_imgs, (0, 2, 3, 1))
Y = np.zeros(X.shape, dtype='float32')
Z = np.zeros(X.shape, dtype='float32')
Y = Z = np.zeros(X.shape, dtype='uint8')
for j in range(Z.shape[0]):
    for i in range(Y.shape[-1]):
        #Y[j, gt_bboxs[j, i, 0, 0]:gt_bboxs[j, i, 0, 2], 
        #  gt_bboxs[j, i, 0, 1]:gt_bboxs[j, i, 0, 3], i] = 1
        Z[j, pred_bboxs[j, i, 0, 0]:pred_bboxs[j, i, 0, 2], 
            pred_bboxs[j, i, 0, 1]:pred_bboxs[j, i, 0, 3], i] = 1
              
Y = np.transpose(pred_masks[np.newaxis, ...], (0, 2, 3, 1))
  
        
X = np.transpose(test_imgs[0], (1, 2, 0))[np.newaxis, ...]
Z = Y = np.transpose(test_labs[0], (1, 2, 0))[np.newaxis, ...]

Y = np.transpose(gt_masks, (1, 2, 0))[np.newaxis, ...]
Z = np.round(pred_masks).astype('uint8')
Z = np.transpose(Z, (1, 2, 0))[np.newaxis, ...]


X = np.transpose(xrays_semseg[0:], (1, 2, 0))[np.newaxis, ...]
Z = Y = np.zeros(X.shape, dtype='float32')

X = np.transpose(test_imgs[20], (1, 2, 0))[np.newaxis, ...]
Y = np.transpose(gt_masks, (1, 2, 0))[np.newaxis, ...]
Z = np.transpose(np.round(pred_masks), (1, 2, 0)).astype('uint8')[np.newaxis, ...]

X = np.transpose(aux1, (0, 2, 3, 1))
Z = Y = np.transpose(test_labs, (0, 2, 3, 1))
          

Z = Y = np.transpose(aux1, (3, 1, 2, 0))
X = np.transpose(aux2, (3, 1, 2, 0))
X = np.transpose(aux9, (3, 1, 2, 0))


X = np.transpose(inputs[1][..., 0:1], (3, 1, 2, 0))
Z = Y = np.transpose(inputs[8], (3, 1, 2, 0))

X = np.transpose(test_imgs[0:1], (0, 2, 3, 1))
Y = np.transpose(gt_masks[np.newaxis, ...], (0, 2, 3, 1))
Z = np.transpose(pred_masks[np.newaxis, ...], (0, 2, 3, 1))




fig, ax = plt.subplots(1, 1)
extent = (0, X.shape[2], 0, X.shape[1])
tracker1 = track.IndexTracker(ax, X, Y, Z, extent, vmin = 0, vmax =1)

fig.canvas.mpl_connect('key_press_event', tracker1.onpress)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress2)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress3)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress4)
fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)   
   




# ----------------------------------------------------------------------------------------
#-----------------------------TEST ON X-RAYS----------------------------------------------

data_path = 'D:/MasterAIThesis/h5py_data/vumc patient data/'
patient_name = 'patient2'
xray_data_path = data_path + patient_name + '/X_rays/'

xrays = dtn.load_data(xray_data_path + 'patient2_x_rays_1.h5', 'img')
xrays = dtn.load_data(xray_data_path + 'patient1_x_rays_clear.h5', 'img')

imgs = imgs.astype('float32')
angles = dtn.load_data(xray_data_path + 'patient2_x_rays_1.h5', 'rot')

np.min(xrays[xrays > 0])
xrays = np.log(xrays + 1)
xrays = (xrays - np.min(xrays))/(np.max(xrays)-np.min(xrays))
xrays = xrays*2 - 1

xrays = utils.log_transform(xrays)

shape = np.array(xrays.shape[1:])
new_shape = shape//2
xrays = xrays[:, shape[0]//2-new_shape[0]//2:shape[0]//2+new_shape[0]//2, 
                shape[1]//2-new_shape[1]//2:shape[1]//2+new_shape[1]//2] 

xrays = (xrays-np.min(xrays))/(np.max(xrays)-np.min(xrays))
xrays = utils.gamma_transform(xrays, 5)
        
# data normalization
mean = np.mean(xrays)
std = np.std(xrays)
xrays = (xrays - mean) / std  




xrays_semseg = xrays_c[..., 64:-64]


outputs_xrays = model.test(xrays_semseg[0:], labs=None, verbose = 0)
outputs_drrs = model.test(test_imgs[3], labs=test_labs[3], verbose = 0)

pred_masks = np.round(outputs_xrays[1]['pred_masks']).astype('uint8')
pred_centers = np.round(outputs_xrays[1]['pred_centroid']).astype('int32')
pred_bboxs = np.concatenate((pred_centers-32, pred_centers+32), axis = 1)
pred_bboxs = pred_bboxs[np.newaxis, :, np.newaxis, :]


image_center = np.array(xrays_semseg[0].shape)[np.newaxis,...]//2           
        
dev_from_center = (pred_centers - image_center)/4
dev_from_center = - dev_from_center

#-----------------------------------------------------------------------

# PLOT X-RAY TRAJECTORY
y_si_semseg = dev_from_center[:, 0]
dev_from_center = - dev_from_center
y_lr_semseg = dev_from_center[:, 1]


"""


