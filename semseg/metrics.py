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

from MaskRCNN import utils
from sklearn.metrics import matthews_corrcoef

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')




def get_box_center(boxes):
    boxes = np.squeeze(boxes, axis = 1)
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    
    z_center = boxes[:, 0] + height // 2
    x_center = boxes[:, 1] + width // 2
    
    return z_center, x_center
    


def ccc(y_true, y_pred):
    cor=np.corrcoef(y_true,y_pred)[0][1]
    
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)
    
    var_true=np.var(y_true)
    var_pred=np.var(y_pred)
    
    sd_true=np.std(y_true)
    sd_pred=np.std(y_pred)
    
    numerator=2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator


def concordance_correlation_coefficient(gt_boxes, pred_centroids):   
    gt_z_center, gt_x_center = get_box_center(gt_boxes)
    pred_z_center, pred_x_center = pred_centroids[:, 0], pred_centroids[:, 1]  
    
    cor_z = ccc(gt_z_center, pred_z_center)
    cor_x = ccc(gt_x_center, pred_x_center)  

    return cor_z, cor_x


def mad(gt_boxes, pred_centroids):
    gt_z_center, gt_x_center = get_box_center(gt_boxes)
    pred_z_center, pred_x_center = pred_centroids[:, 0], pred_centroids[:, 1]  
    
    z_mad = np.mean(np.abs(gt_z_center - pred_z_center))
    z_std = np.std(np.abs(gt_z_center - pred_z_center))
    
    x_mad = np.mean(np.abs(gt_x_center - pred_x_center))
    x_std = np.std(np.abs(gt_x_center - pred_x_center))
    
    return z_mad, z_std, x_mad, x_std
    
    
    
def ad(gt_boxes, pred_boxes):
    gt_z_center, gt_x_center = get_box_center(gt_boxes)
    pred_z_center, pred_x_center = get_box_center(pred_boxes) 
    
    x_ad = np.abs(gt_x_center - pred_x_center)
    z_ad = np.abs(gt_z_center - pred_z_center)
    
    return z_ad, x_ad
    
    
       
    
    
    
    
    
    
    

