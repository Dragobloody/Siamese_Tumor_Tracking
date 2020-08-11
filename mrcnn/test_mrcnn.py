import os
import numpy as np
import keras.layers
import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn.metrics import mean_squared_error
import h5py
import scipy
import skimage
import random
import csv
import cv2
import index_tracker as track  
from keras.callbacks import LearningRateScheduler
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# set seed
from numpy.random import seed
seed(1234)
from tensorflow import set_random_seed
set_random_seed(1234)


from MaskRCNN import maskrcnn_model as modellib, data_generator as dg, config as conf, visualize, utils, metrics


def load_data(data_path, data_name):
    # load data    
    f = h5py.File(data_path,"r")
    data = f[data_name][:]
    f.close()
    return data

def save_data(data_path, file_name, data):
    # save data    
    f_name = data_path + file_name + ".hdf5"
    f = h5py.File(f_name)
    f.create_dataset(file_name, data = data)
    f.close()


def ct_resize(data, zoom, cval=0.0):
	"""
		dataset - Input ct scans, must be a 4d numpy array 
		zoom - Zoom factor , float or array of floats for each dimensions
		cval - Value used for points outside the boundaries of the input
	"""

	out_data = scipy.ndimage.interpolation.zoom(data, zoom = zoom, cval=cval)

	return out_data

def resize_all(dataset, zoom, cval = 0.0):
	"""
		dataset - Input ct scans, must be a 4d numpy array 
		zoom - Zoom factor , float or array of floats for each dimensions
		cval - Value used for points outside the boundaries of the input
	"""
	return np.stack([ct_resize(dataset[i], zoom, cval) for i in range(dataset.shape[0])], axis = 0)


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)
    
    
# fit an ellipse to segmentation mask
def fit_ellipse(mask):
    mask = scipy.ndimage.binary_fill_holes(mask).astype('uint8')
    non_zero_points = cv2.findNonZero(mask)
    elps = cv2.fitEllipse(non_zero_points)
    result = cv2.ellipse(np.zeros(mask.shape, 'uint8'), elps, (1, 1, 1), -1)
    
    return result


def test_patient(patient_name, test_imgs, test_labs, test_model, config, writer, angle):    
    # add new axis
    test_imgs = test_imgs[..., np.newaxis]
    test_labs = test_labs[..., np.newaxis]
    
    # Create model in training mode    
    gt, pred, nr_pred = test_model.test(test_imgs, test_labs, verbose=0)    
    # get groud truth data
    gt_bboxs = gt['gt_bboxs']
    gt_class_ids =  gt['gt_class_ids']
    gt_masks =  gt['gt_masks']
    # get predicted data
    pred_bboxs =  pred['pred_bboxs'][:, np.newaxis, ...]
    pred_class_ids =  pred['pred_class_ids'][:, np.newaxis, ...]
    pred_scores =  pred['pred_scores'][:, np.newaxis, ...]
    pred_masks =  pred['pred_masks'][..., np.newaxis]
    
    # metrics
    accuracy = metrics.accuracy(gt_bboxs, pred_bboxs, iou_threshold = 0.7)
    cor_z, cor_x = metrics.concordance_correlation_coefficient(gt_bboxs, pred_bboxs)
    z_mad, z_std, x_mad, x_std = metrics.mad(gt_bboxs, pred_bboxs)
    
    # save metrics
    writer.writerow({'Patient':patient_name, 'Angle':angle, 'Nr_Predictions':nr_pred, 'Acc':accuracy, 'Cor_z':cor_z, 'Cor_x':cor_x,
                     'MAD_z':z_mad/3, 'STD_z':z_std/3, 'MAD_x':x_mad/3, 'STD_x':x_std/3})       
        
    
    
"""
    
    
# config
config = conf.Config()
config.NAME = 'lung_tumor'
config.BACKBONE = "resnet50"
config.BATCH_SIZE =  1
config.IMAGES_PER_GPU = 1
TRAIN_BN = True
config.IMAGE_HEIGHT = 384
config.IMAGE_WIDTH = 384
config.IMAGE_SHAPE = [384, 384, 1]
config.MASK = False
config.FPN_CLASSIF_FC_LAYERS_SIZE = 256
config.DETECTION_MIN_CONFIDENCE = 0.1

data_path = 'D:/MasterAIThesis/h5py_data/vumc patient data/'   
model = 'mrcnn'
data_mode = 'standard'
# compute metrics

# load test data
patient_name = 'patient1'
model_path = data_path + patient_name + '/models/' + model + '/' + data_mode
drr_data_path = model_path + '/dataset/'    
LOG_DIR = model_path + '/logs/'

imgs = load_data(drr_data_path + 'train_imgs.hdf5', 'train_imgs')
labs = load_data(drr_data_path + 'train_labs.hdf5', 'train_labs')
test_imgs = load_data(drr_data_path + 'test_imgs.hdf5', 'test_imgs')
test_labs = load_data(drr_data_path + 'test_labs.hdf5', 'test_labs')

# get the center 384x384 region
imgs = imgs[..., 64:-64]
labs = labs[..., 64:-64]
test_imgs = test_imgs[..., 64:-64]
test_labs = test_labs[..., 64:-64]

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
imgs = np.reshape(imgs, (-1, s[2], s[3]))
labs = np.reshape(labs, (-1, s[2], s[3]))
test_imgs = np.reshape(test_imgs, (-1, s[2], s[3]))
test_labs = np.reshape(test_labs, (-1, s[2], s[3]))

# load model
model = modellib.MaskRCNN(mode="training", model_dir=LOG_DIR,
                          config=config)    

weights_path = model.find_last()
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True) 

model.keras_model.summary()


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




  
data_generator = modellib.data_generator(test_imgs, test_labs, 
                                         config, 2, np.array([1]), 
                                         shuffle=True, 
                                         batch_size=config.BATCH_SIZE)
start = time.time()
inputs, _ = next(data_generator)
stop = time.time()
print(stop - start)

start = time.time()
outputs = model.keras_model.predict(inputs)
stop = time.time()
print(stop - start)


# TEST MODEL
test_imgs_mrcnn = test_imgs[..., 64:-64]
test_labs_mrcnn = test_labs[..., 64:-64]
s = test_imgs_mrcnn.shape
test_imgs_mrcnn = np.reshape(test_imgs_mrcnn, (-1, s[2], s[3]))
test_labs_mrcnn = np.reshape(test_labs_mrcnn, (-1, s[2], s[3]))


# load model
test_model = modellib.MaskRCNN(mode="inference", model_dir=LOG_DIR,
                          config=config)    
weights_path = test_model.find_last()
# Load weights
print("Loading weights ", weights_path)
test_model.load_weights(weights_path, by_name=True)    




gt, pred, nr_pred = test_model.test(test_imgs_mrcnn, test_labs_mrcnn, verbose=0)    
# get groud truth data
gt_bboxs = gt['gt_bboxs']
gt_class_ids =  gt['gt_class_ids']
gt_masks =  gt['gt_masks']
# get predicted data
pred_bboxs =  pred['pred_bboxs'][:, np.newaxis, ...]
pred_class_ids =  pred['pred_class_ids'][:, np.newaxis, ...]
pred_scores =  pred['pred_scores'][:, np.newaxis, ...]
#pred_masks =  pred['pred_masks'][..., np.newaxis]


miou = metrics.mIOU(gt_bboxs, pred_bboxs)
accuracy = metrics.accuracy(gt_bboxs, pred_bboxs, 
                            iou_threshold = 0.7)
cor_z, cor_x = metrics.concordance_correlation_coefficient(gt_bboxs, pred_bboxs)
z_mad, z_std, x_mad, x_std = metrics.mad(gt_bboxs, pred_bboxs)    
z_mad, z_std, x_mad, x_std = z_mad/3, z_std/3, x_mad/3, x_std/3 


z_ad_mrcnn, x_ad_mrcnn = metrics.ad(gt_bboxs, pred_bboxs)




gt_masks = utils.bboxs_to_masks(gt_bboxs)
pred_masks = utils.bboxs_to_masks(pred_bboxs)
gt_masks = gt_masks[:, 0, :, :]
pred_masks = pred_masks[:, 0, :, :]



data_path = 'D:/MasterAIThesis/h5py_data/vumc patient data/'
patient_name = 'patient2'
xray_data_path = data_path + patient_name + '/X_rays/'

xrays = load_data(xray_data_path + 'patient2_x_rays_1.h5', 'img')
angles = load_data(xray_data_path + 'patient2_x_rays_1.h5', 'rot')
angles = angles-270


xrays = utils.log_transform(xrays)

shape = np.array(xrays.shape[1:])
new_shape = [384, 384]
xrays = xrays[:, shape[0]//2-new_shape[0]//2:shape[0]//2+new_shape[0]//2, 
                shape[1]//2-new_shape[1]//2:shape[1]//2+new_shape[1]//2] 

xrays = (xrays-np.min(xrays))/(np.max(xrays)-np.min(xrays))
xrays = utils.gamma_transform(xrays, 5)

xrays_mrcnn = xrays[..., 64:-64, np.newaxis]

nr_pred = 0
pred_bboxs, pred_class_ids, pred_masks, pred_scores = [], [], [], [] 
start = time.time()

for i in range(xrays_mrcnn.shape[0]):  
    result = test_model.detect([xrays_mrcnn[i]], 0)
    if len(result[0]['scores']) > 0:
        max_idx = np.argmax(result[0]['scores'])
        pred_bboxs.append(result[0]['rois'][max_idx])
        pred_class_ids.append(result[0]['class_ids'][max_idx])
        pred_scores.append(result[0]['scores'][max_idx])
        #pred_masks.append(result[0]['masks'][..., max_idx])
        nr_pred += 1
        print(pred_bboxs[-1])
    else:
        pred_bboxs.append(pred_bboxs[-1])
        pred_class_ids.append(pred_class_ids[-1])
        pred_scores.append(0)
        print(pred_bboxs[-1])
        
stop = time.time()
print(stop-start)

nr_pred = nr_pred/xrays_mrcnn.shape[0]
 
pred_bboxs = np.stack(pred_bboxs, axis = 0)
pred_class_ids = np.stack(pred_class_ids, axis = 0)
pred_scores = np.stack(pred_scores, axis = 0)
#pred_masks = np.stack(pred_masks, axis = 0)

#-----------------------------------------------------------------------


gt_masks = utils.bboxs_to_masks(gt_bboxs)
pred_masks = utils.bboxs_to_masks(pred_bboxs)
gt_masks = gt_masks[:, 0, :, :]
pred_masks = pred_masks[:, 0, :, :]



X = np.transpose(imgs[np.newaxis,...], (0, 2, 3, 1))
Z = Y = np.transpose(labs[np.newaxis,...], (0, 2, 3, 1))

X = np.transpose(xrays, (3, 1, 2, 0))
Z = Y = np.transpose(pred_masks[np.newaxis,...], (0, 2, 3, 1))

X = np.transpose(test_imgs[np.newaxis,...], (0, 2, 3, 1))
Y = np.transpose(test_labs[np.newaxis,...], (0, 2, 3, 1))
Z = np.transpose(pred_masks[...,0][np.newaxis,...], (0, 2, 3, 1))
Z = np.transpose(pred_masks[np.newaxis,...], (0, 2, 3, 1))


X = np.transpose(pred_masks[np.newaxis,...], (0, 2, 3, 1)).astype('float32')
Z = Y = np.zeros(X.shape, dtype='uint8')

fig, ax = plt.subplots(1, 1)
extent = (0, X.shape[2], 0, X.shape[1])
tracker1 = track.IndexTracker(ax, X, Y, Z, extent, vmin = 0, vmax =1)

fig.canvas.mpl_connect('key_press_event', tracker1.onpress)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress2)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress3)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress4)
fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)   
   

#-----------------------------------------------------------------------------

pred_centers =  np.array([pred_bboxs[..., 0] + (pred_bboxs[..., 2] -  pred_bboxs[..., 0])//2,  pred_bboxs[..., 1] + (pred_bboxs[..., 3] -  pred_bboxs[..., 1])//2])    
pred_centers = np.transpose(pred_centers, (1, 0)) 

image_center = np.array(xrays[0,...,0].shape)[np.newaxis,...]//2   
        
dev_from_center = (pred_centers - image_center)/4
dev_from_center = - dev_from_center

y_si_mrcnn = dev_from_center[:, 0]
dev_from_center = - dev_from_center
y_lr_mrcnn = dev_from_center[:, 1]

"""