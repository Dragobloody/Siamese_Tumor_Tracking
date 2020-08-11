"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import numpy as np
import tensorflow as tf
import keras
import scipy
import time 
from semseg import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Data Generator
############################################################

def load_image_gt(config, input_image, input_mask=None, mode = 'training'):
    """Load and return ground truth data for an image (image, mask, bounding boxes).   

    Returns:
    image: [height, width, 1]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask  
    if len(input_image.shape) < 3:
        input_image = input_image[..., np.newaxis]
        
    if input_mask is not None:
        if len(input_mask.shape) < 3:
            input_mask = input_mask[..., np.newaxis]     
    
    # Flip image in training mode
    if mode == 'training':
        flip = random.randint(0, 1)
        if flip == 1:
            input_image = np.flip(input_image, axis=1)
            input_mask = np.flip(input_mask, axis=1)          
                     
        # random contrast transform
        gamma = random.uniform(0.3, 2)
        input_image = utils.gamma_transform(input_image, gamma) 
        
        # shift
        shift_x = random.randint(-32, 32)
        shift_y = random.randint(-32, 32)   

        input_image = scipy.ndimage.shift(input_image, (shift_x, shift_y, 0), order=0)
        input_mask = scipy.ndimage.shift(input_mask, (shift_x, shift_y, 0), order=0)
       
    else:
        # shift
        shift_x = shift_y = 0
        #shift_x = random.randint(-8, 8)
        #shift_y = random.randint(-8, 8)        
        #input_image = scipy.ndimage.shift(input_image, (shift_x, shift_y, 0), order=0)
        #input_mask = scipy.ndimage.shift(input_mask, (shift_x, shift_y, 0), order=0)
     
    gt_bbox = None
    if input_mask is not None:    
        gt_bbox = utils.extract_bboxes(input_mask)    
    
    return input_image, gt_bbox, input_mask


