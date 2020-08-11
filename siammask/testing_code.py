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
import  keras
import keras.backend as K
#K.set_floatx('float32')
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

from siammask import utils, rpn as rp, data_generator as dg, visualize, metrics
from siammask.config import Config


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


"""
class TumorConfig(Config):    
    NAME = 'lung_tumor'    
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,   
        "rpn_centerness_loss": 1.,        
        "rpn_mask_loss": 10.        
    }
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    IMAGES_PER_GPU = 1
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
    BBOX_LOSS_LAMDA_D = 10
    BBOX_LOSS_LAMDA_C = 1
    
    TRAIN_MASKS_PER_IMAGE = 20
    
    USE_BN = True
    USE_DP = False
    
    ANCHORLESS = True 
    SIAMRPN = True 
    CROP = True
    
    MASK = False
    REFINE = False    
    STAGE4 = True
    SHARED = False
    BACKBONE = "resnet50"
    
    BBOX_RATIO = 0.7
    
    BBOX_SIGMA = 1.0
    
    KERNEL_IMAGE_SIZE = 128
    IMAGE_CHANNEL_COUNT = 3
    if CROP:
        FEATURES_SHAPE = [25, 25]
    else:
        FEATURES_SHAPE = [17, 17]
    


# SET UP CONFIG
config = TumorConfig()

data_path = 'D:/MasterAIThesis/h5py_data/vumc phantom data/'
patient_name = 'phantom3'
model = 'siammask'
data_mode = 'decorrelated'
drr_data_path = data_path + patient_name + '/models/'+ model + '/' + data_mode +'/dataset/'
# Directory to save logs and trained model
LOG_DIR =  data_path + patient_name + '/models/' + model + '/' + data_mode +'/logs/'


gans = '_srcygans'
gans=''
imgs = dtn.load_data(drr_data_path + 'train_imgs'+gans+'.hdf5', 'train_imgs'+gans)
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

imgs = (imgs - np.min(imgs))/(np.max(imgs)-np.min(imgs))
test_imgs = (test_imgs - np.min(test_imgs))/(np.max(test_imgs)-np.min(test_imgs))

# data normalization
mean = np.mean(imgs)
std = np.std(imgs)
imgs = (imgs - mean) / std 
test_imgs = (test_imgs - mean) / std 


    
# clear memory 1
K.clear_session()   
from numba import cuda
cuda.select_device(0)
cuda.close()
     

model = modellib.SiamMask(mode="training", config=config,
                                      model_dir=LOG_DIR)
resnet_weights_path = 'D:/MasterAIThesis/code/siammask/pretrained_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
coco_weights_path = 'D:/MasterAIThesis/code/siammask/pretrained_weights/mask_rcnn_coco.h5'


model.load_weights(resnet_weights_path, by_name=True, exclude='4+') 
model.load_weights(coco_weights_path, by_name=True, exclude='4+') 


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

def lr_scheduler_2(epoch, lr):
    if epoch > 0  and lr > 1e-5: 
        k = 0.03
        lr = lr*np.exp(-k*epoch)    
    return lr
lrs = LearningRateScheduler(lr_scheduler)  

model.train(imgs, labs, 
            test_imgs, test_labs,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            layers='all',
            custom_callbacks = [lrs])


data_generator_drrs = modellib.data_generator(test_imgs, test_labs, config, 2, [1], shuffle=False, batch_size=config.BATCH_SIZE, mode = model.mode)

inputs_drrs, _ = next(data_generator_drrs)
inputs_xrays, _ = next(data_generator_xrays)

start = time.time()
inputs_drrs, _ = next(data_generator_drrs)
outputs = model.keras_model.predict(inputs_drrs)
stop = time.time()
print(stop - start)

start = time.time()
inputs_drrs, _ = next(data_generator_drrs)
outputs = model.keras_model.predict(inputs_drrs[0:4])
stop = time.time()
print(stop - start)

outputs = model.keras_model.predict(inputs_xrays[0:4])
outputs = model.detect(inputs_drrs[0], inputs_drrs[1], inputs_drrs[2])

accs = []

start = time.time()        
outputs = model.test(test_imgs[0, :], labs = test_labs[0, :], verbose = 0)
stop = time.time()
print(stop-start)
gt_bboxs = outputs[0]['gt_bboxs']
pred_bboxs = outputs[1]['pred_bboxs'][:, np.newaxis, :]
gt_search_bboxs = outputs[0]['gt_search_bboxs']
pred_search_bboxs = outputs[1]['pred_search_bboxs'][:, np.newaxis, :]
gt_masks = outputs[0]['gt_masks']
pred_masks = outputs[1]['pred_masks']
pred_masks = np.round(pred_masks)


miou = metrics.mIOU(gt_bboxs, pred_bboxs)
accuracy = metrics.accuracy(gt_bboxs, pred_bboxs, 
                            iou_threshold = 0.7)
accs.append(accuracy)
cor_z, cor_x = metrics.concordance_correlation_coefficient(gt_search_bboxs, pred_search_bboxs)
z_mad, z_std, x_mad, x_std = metrics.mad(gt_bboxs, pred_bboxs)    
z_mad, z_std, x_mad, x_std = z_mad/3, z_std/3, x_mad/3, x_std/3 

z_ad_anchor_free, x_ad_anchor_free = metrics.ad(gt_bboxs, pred_bboxs)
z_ad_anchor_based, x_ad_anchor_based = metrics.ad(gt_bboxs, pred_bboxs)
z_ad_anchor_free_mask, x_ad_anchor_free_mask = metrics.ad(gt_bboxs, pred_bboxs)




wilcox_anchor_free_mrcnn = scipy.stats.wilcoxon(z_ad_anchor_free, z_ad_mrcnn, alternative='less')


# simulate test data
gt_bboxs, pred_bboxs = [], []      
gt_search_bboxs, pred_search_bboxs = [], []  
gt_masks, pred_masks = [], []   
 
for i in range(test_imgs.shape[0]):
    gt, pred = model.test(test_imgs[i], test_labs[i], verbose = 0)
    gt_bboxs.append(gt['gt_bboxs'])
    pred_bboxs.append(pred['pred_bboxs'][:, np.newaxis, :])
    gt_search_bboxs.append(gt['gt_search_bboxs'])
    pred_search_bboxs.append(pred['pred_search_bboxs'][:, np.newaxis, :])
    gt_masks.append(gt['gt_masks'])
    pred_masks.append(pred['pred_masks'])
gt_bboxs = np.stack(gt_bboxs, axis = 0)
pred_bboxs = np.stack(pred_bboxs, axis = 0)
gt_search_bboxs = np.stack(gt_search_bboxs, axis = 0)
pred_search_bboxs = np.stack(pred_search_bboxs, axis = 0)
gt_masks = np.stack(gt_masks, axis = 0)
pred_masks = np.stack(pred_masks, axis = 0)


gt_bboxs = np.moveaxis(gt_bboxs, 1, 0)
pred_bboxs = np.moveaxis(pred_bboxs, 1, 0)
gt_bboxs = np.reshape(gt_bboxs[:-1, ...], (9, 20, gt_bboxs.shape[1], gt_bboxs.shape[2], gt_bboxs.shape[3]))
pred_bboxs = np.reshape(pred_bboxs[:-1, ...], (9, 20, pred_bboxs.shape[1], pred_bboxs.shape[2], pred_bboxs.shape[3]))         
gt_bboxs = np.reshape(gt_bboxs, (gt_bboxs.shape[0], gt_bboxs.shape[1]*gt_bboxs.shape[2], gt_bboxs.shape[3], gt_bboxs.shape[4]))
pred_bboxs = np.reshape(pred_bboxs, (pred_bboxs.shape[0], pred_bboxs.shape[1]*pred_bboxs.shape[2], pred_bboxs.shape[3], pred_bboxs.shape[4]))
   

gt_search_bboxs = np.moveaxis(gt_search_bboxs, 1, 0)
pred_search_bboxs = np.moveaxis(pred_search_bboxs, 1, 0)
gt_search_bboxs = np.reshape(gt_search_bboxs[:-1, ...], (9, 20, gt_search_bboxs.shape[1], gt_search_bboxs.shape[2], gt_search_bboxs.shape[3]))
pred_search_bboxs = np.reshape(pred_search_bboxs[:-1, ...], (9, 20, pred_search_bboxs.shape[1], pred_search_bboxs.shape[2], pred_search_bboxs.shape[3]))         
gt_search_bboxs = np.reshape(gt_search_bboxs, (gt_search_bboxs.shape[0], gt_search_bboxs.shape[1]*gt_search_bboxs.shape[2], gt_search_bboxs.shape[3], gt_search_bboxs.shape[4]))
pred_search_bboxs = np.reshape(pred_search_bboxs, (pred_search_bboxs.shape[0], pred_search_bboxs.shape[1]*pred_search_bboxs.shape[2], pred_search_bboxs.shape[3], pred_search_bboxs.shape[4]))
 
gt_bboxs = np.reshape(gt_bboxs, (-1, 1, 4))
pred_bboxs = np.reshape(pred_bboxs, (-1, 1, 4)) 
gt_search_bboxs = np.reshape(gt_search_bboxs, (-1, 1, 4))
pred_search_bboxs = np.reshape(pred_search_bboxs, (-1, 1, 4))
gt_masks = np.reshape(gt_masks, (-1, 384, 512))
pred_masks = np.reshape(pred_masks, (-1, 384, 512))

# open csv file
csv_file = data_path + patient_name + '/models/'+ test_model + '/' + test_data_mode + '/results/metrics1.csv'
f = open(csv_file, 'w', newline = '')
fieldnames = ['Patient', 'Angle', 'Nr_Predictions', 'mIOU', 'AP50', 'AP70', 'Cor_z', 'Cor_x', 'MAD_z', 'STD_z', 'MAD_x', 'STD_x']
writer = csv.DictWriter(f, fieldnames = fieldnames)
writer.writeheader()

for j in range(gt_bboxs.shape[0]):
    print(j)
    angle = str(j*20) + '_' + str((j+1)*20)
    # metrics
    miou = metrics.mIOU(gt_bboxs[j], pred_bboxs[j])
    ap70 = metrics.accuracy(gt_bboxs[j], pred_bboxs[j], iou_threshold = 0.7)
    cor_z, cor_x = metrics.concordance_correlation_coefficient(gt_search_bboxs[j], pred_search_bboxs[j])
    z_mad, z_std, x_mad, x_std = metrics.mad(gt_bboxs[j], pred_bboxs[j])        
    # save metrics
    writer.writerow({'Patient':patient_name, 'Angle':angle, 'Nr_Predictions':1, 
                     'mIOU':miou, 'AP70':ap70, 'Cor_z':cor_z, 'Cor_x':cor_x,
                     'MAD_z':z_mad/3, 'STD_z':z_std/3, 'MAD_x':x_mad/3, 'STD_x':x_std/3}) 
    
f.close()

gt_masks = utils.bboxs_to_masks(gt_bboxs)
pred_masks = utils.bboxs_to_masks(pred_bboxs)
gt_masks = gt_masks[:, 0, :, :]
pred_masks = pred_masks[:, 0, :, :]

X = outputs[4]
Y = Z = np.zeros(X.shape, dtype='uint8')


X = np.transpose(test_imgs, (0, 2, 3, 1))
Z = Y = np.transpose(test_labs, (0, 2, 3, 1))

Z = Y = np.zeros(X.shape, dtype='float32')


Y = Z = np.zeros(X.shape, dtype='uint8')
for j in range(Z.shape[0]):
    for i in range(Y.shape[-1]):
        #Y[j, gt_bboxs[j, i, 0, 0]:gt_bboxs[j, i, 0, 2], 
        #  gt_bboxs[j, i, 0, 1]:gt_bboxs[j, i, 0, 3], i] = 1
        if pred_scores[i, 0] >= 0.01:
            Z[j, pred_bboxs[j, i, 0, 0]:pred_bboxs[j, i, 0, 2], 
              pred_bboxs[j, i, 0, 1]:pred_bboxs[j, i, 0, 3], i] = 1
              
 

X = np.transpose(xrays_c[0:], (1, 2, 0))[np.newaxis, ...]
Z = Y = np.zeros(X.shape, dtype='float32')


X = np.transpose(test_imgs[0:1, :], (0, 2, 3, 1))
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

data_path = 'D:/MasterAIThesis/h5py_data/vumc phantom data/'
patient_name = 'phantom3'
xray_data_path = data_path + patient_name + '/X_rays/'

xrays = dtn.load_data(xray_data_path + 'phantom3_x_rays_1.h5', 'img')
xrays = dtn.load_data(xray_data_path + 'phantom3_x_rays_2.h5', 'img')



xrays = dtn.load_data(xray_data_path + 'patient2_x_rays_1.h5', 'img')
xrays = dtn.load_data(xray_data_path + 'patient2_x_rays_2.h5', 'img')

xrays = dtn.load_data(xray_data_path + 'patient1_x_rays_2.h5', 'img')

imgs = imgs.astype('float32')
angles = dtn.load_data(xray_data_path + 'phantom3_x_rays_2.h5', 'rot')
angles = angles-90
angles.sort()

angles = dtn.load_data(xray_data_path + 'patient1_x_rays_2.h5', 'rot')
angles = angles-270
angles.sort()



xrays = utils.log_transform(xrays)
xrays2 = utils.log_transform(xrays2)

shape = np.array(xrays.shape[1:])
new_shape = shape//2
xrays = xrays[:, shape[0]//2-new_shape[0]//2:shape[0]//2+new_shape[0]//2, 
                shape[1]//2-new_shape[1]//2:shape[1]//2+new_shape[1]//2] 

xrays = (xrays + 1)/2 
xrays2 = (xrays2 + 1)/2 

xrays = (xrays-np.min(xrays))/(np.max(xrays)-np.min(xrays))

xrays = utils.gamma_transform(xrays, 5)
xrays_c =  utils.gamma_transform(xrays, 9) 

imgs_c = utils.gamma_transform(imgs, 0.2) 
     
# data normalization
mean = np.mean(xrays_c)
std = np.std(xrays_c)
xrays_c = (xrays_c - mean) / std  





outputs_xrays = model.test(xrays_c[0:], labs=None, verbose = 0)
outputs_drrs = model.test(test_imgs[3], labs=test_labs[3], verbose = 0)

pred_masks = np.round(outputs_xrays[1]['pred_masks']).astype('uint8')
pred_bboxs = outputs_xrays[1]['pred_bboxs'][np.newaxis, :, np.newaxis, :]
pred_scores = outputs_xrays[1]['pred_scores']

pred_centers =  np.array([pred_bboxs[..., 0] + (pred_bboxs[..., 2] -  pred_bboxs[..., 0])//2,  pred_bboxs[..., 1] + (pred_bboxs[..., 3] -  pred_bboxs[..., 1])//2])    
pred_centers = np.transpose(pred_centers[:, 0, :, 0], (1, 0)) 

image_center = np.array(xrays[0].shape)[np.newaxis,...]//2   
        
dev_from_center = (pred_centers - image_center)/4
dev_from_center = - dev_from_center

#-----------------------------------------------------------------------

# PLOT X-RAY TRAJECTORY

y_si_srgans = dev_from_center[:, 0]
dev_from_center = - dev_from_center
y_lr_srgans = dev_from_center[:, 1]


y_si_anchor_free = dev_from_center[:, 0]
dev_from_center = - dev_from_center
y_lr_anchor_free = dev_from_center[:, 1]


def plot_trajectory(ax, angles, y, model_name, direction='SI', color='blue', alpha=0.6, lim=10, angle_interval=20, max_angle=180):
    x = angles
    xvlines = np.arange(0, max_angle+1, angle_interval)
    xhlines = np.arange(-lim, lim+1, 2)

    ax.set_facecolor('lightgrey')
    ax.plot(x, y, label=model_name, color=color, alpha=alpha)
    ax.set_ylim([-lim, lim])
    ax.set_xticks(np.arange(0, max_angle+1, angle_interval))
    ax.set_yticks(np.arange(-lim, lim+1, 2))
    ax.set_xlabel('Angle (degree)')
    ax.set_ylabel('Position ('+direction+' direction) (mm)')    
    
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, ncol=3)
    for xv in xvlines:
        ax.axvline(x=xv, linewidth = 0.5, color=(0,0,0), linestyle = 'dashed')
    for xh in xhlines:
        ax.axhline(y=xh, linewidth = 0.5, color=(0,0,0), linestyle = 'dashed')
    ax.axhline(y=0, linewidth = 1, color=(0,0,0))
    
    return ax


f, ax1 = plt.subplots(1, figsize=(20, 4))
# SI siamese-mrcnn-semseg
ax1 = plot_trajectory(ax1, angles, y_si_mrcnn, model_name='Zhao et al. [8]', direction='SI', color='blue', max_angle=180)
ax1 = plot_trajectory(ax1, angles, y_si_semseg, model_name='Takahashi et al. [10]', direction='SI', color='green', max_angle=140)
ax1 = plot_trajectory(ax1, angles, y_si_anchor_free, model_name='Siamese model',  direction='SI', color='red', alpha=0.7, max_angle=140)

# LR siamese-mrcnn-semseg
ax1 = plot_trajectory(ax1, angles, y_lr_mrcnn, model_name='Zhao et al. [8]', direction='LR', color='blue', max_angle=140)
ax1 = plot_trajectory(ax1, angles, y_lr_semseg, model_name='Takahashi et al. [10]', direction='LR', color='green', max_angle=140)
ax1 = plot_trajectory(ax1, angles, y_lr_anchor_free, model_name='Siamese model',  direction='LR', color='red', alpha=0.7, max_angle=140)


# SI anchor-free-based-mask
ax1 = plot_trajectory(ax1, angles, y_si_anchor_based, model_name='anchor-based',  direction='SI', color='green')
ax1 = plot_trajectory(ax1, angles, y_si_anchor_free, model_name='anchor-free',  direction='SI', color='red', alpha=0.7)
ax1 = plot_trajectory(ax1, angles, y_si_anchor_free_mask, model_name='anchor-free + mask', direction='SI', color='blue')

# LR anchor-free-based-mask
ax1 = plot_trajectory(ax1, angles, y_lr_anchor_based, model_name='anchor-based',  direction='LR', color='green')
ax1 = plot_trajectory(ax1, angles, y_lr_anchor_free, model_name='anchor-free',  direction='LR', color='red', alpha=0.7)
ax1 = plot_trajectory(ax1, angles, y_lr_anchor_free_mask, model_name='anchor-free + mask', direction='LR', color='blue')


# SI nogans-srgans-cygans-srcygans
f, ax1 = plt.subplots(1, figsize=(20, 4))
ax1 = plot_trajectory(ax1, angles[:], y_si_no_gans[:], model_name='siamese',  direction='SI', color='red', angle_interval = 40, max_angle = 360)
ax1 = plot_trajectory(ax1, angles[:], y_si_srgans[:], model_name='siamese + SRGAN',  direction='SI', color='green', angle_interval = 40, max_angle = 360)
ax1 = plot_trajectory(ax1, angles[:], y_si_srcygans[:], model_name='siamese + SRGAN + CycleGAN',  direction='SI', color='blue', angle_interval = 40, max_angle = 360)


# LR nogans-srgans-cygans-srcygans

f, ax2 = plt.subplots(1, figsize=(20, 4))
ax1 = plot_trajectory(ax2, angles[:], y_lr_no_gans[:], model_name='siamese',  direction='LR', color='red', lim=16, angle_interval = 40, max_angle = 360)
ax1 = plot_trajectory(ax2, angles[:], y_lr_srgans[:], model_name='siamese + SRGAN',  direction='LR', color='green', lim=16, angle_interval = 40, max_angle = 360)
ax1 = plot_trajectory(ax2, angles[:], y_lr_srcygans[:], model_name='siamese + SRGAN + CycleGAN',  direction='LR', color='blue', lim=16, angle_interval = 40, max_angle = 360)




#------------------------------------------------------------------------
# READ CSV FILES WITH TUMOR TRAJECTORY 
couch_lat, couch_long = [], []
file_loc = "D:/MasterAIThesis/h5py_data/vumc phantom data/phantom3/axes_T3_ExpB_reg.csv"
with open(file_loc) as f:
    read_csv = csv.reader(f, delimiter=',')
    for row in read_csv:
        couch_lat.append(row[17])
        couch_long.append(row[18])

del couch_lat[0]
del couch_long[0]

for i in range(len(couch_lat)):
    couch_lat[i] = float(couch_lat[i])
    couch_long[i] = float(couch_long[i])

couch_lat = np.array(couch_lat)
couch_long = np.array(couch_long)

couch_long = (couch_long - np.min(couch_long))/(np.max(couch_long) - np.min(couch_long)) 
couch_long = couch_long*10 - 6

couch_lat = (couch_lat - np.min(couch_lat))/(np.max(couch_lat) - np.min(couch_lat)) 
couch_lat = couch_lat*10 - 6

f, ax = plt.subplots(1, figsize=(20, 4))
ax = plot_trajectory(ax, angles[:], couch_long[:], model_name='GT',  direction='SI', color='green', lim=10, angle_interval = 40, max_angle = 360)

f, ax = plt.subplots(1, figsize=(20, 4))
ax = plot_trajectory(ax, angles[:], couch_lat[:], model_name='GT',  direction='LR', color='green', lim=16, angle_interval = 40, max_angle = 360)

"""
