"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Loss Functions
############################################################



def smooth_l1_loss(y_true, y_pred, sigma = 1.0):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0/(sigma**2)), "float32")
    loss = (less_than_one * 0.5 * (sigma**2) * (diff**2)) + (1 - less_than_one) * (diff - 1.0/(2*sigma**2))
    
    return loss


def binary_logistic_regression_with_logits(target, output):   
    """ Mask loss. """  
    # transform target from 0/1 to -1/1
    target = target*2 - 1    
    # compute loss
    loss = tf.math.reduce_mean(tf.math.log1p(tf.math.exp(-tf.math.multiply(target, output))), axis = (1,2))
    
    return loss

def binary_logistic_regression(target, output):   
    """ Mask loss. """  
    # transform target from 0/1 to -1/1
    target = target*2 - 1
    # convert from sigmoid probabilities back to logits
    epsilon = tf.convert_to_tensor(1e-7, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1 - epsilon)
    output = tf.log(output / (1 - output))
    # compute loss
    loss = tf.math.reduce_mean(tf.math.log1p(tf.math.exp(-tf.math.multiply(target, output))), axis = (1,2))
    
    return loss



def rpn_mask_loss_graph(target_masks, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    
    pred_masks = tf.squeeze(pred_masks, -1)
    target_masks = tf.squeeze(target_masks, -1) 
    target_masks = tf.cast(target_masks, tf.float32)    
    
    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(target_masks) > 0,
                    K.binary_crossentropy(target=target_masks, output=pred_masks),
                    tf.constant(0.0))
    loss = K.mean(loss)
    
    return loss



