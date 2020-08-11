"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K


#from MaskRCNN import utils, resnet, rpn, proposal, detection, fpn
#from MaskRCNN import config
from focal_loss import binary_focal_loss, BinaryFocalLoss

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


def focal_loss(gamma=2., alpha=.25, from_logits=False):
    def focal_loss_fixed(y_true, y_pred):
        if from_logits:
            y_pred = tf.sigmoid(y_pred)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        # avoid NaN values
        eps = K.epsilon()
        pt_1 = K.clip(pt_1, eps, 1-eps)
        pt_0 = K.clip(pt_0, eps, 1-eps)
        
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    
    return focal_loss_fixed


def CIoULoss(y_true, y_pred, lamda_d = 1, lamda_c = 0, loss_type = "diou"):
    '''
    takes in a list of bounding boxes
    but can work for a single bounding box too
    all the boundary cases such as bounding boxes of size 0 are handled.
    '''    

    x1g, y1g, x2g, y2g = tf.split(value=y_true, num_or_size_splits=4, axis=1)
    x1, y1, x2, y2 = tf.split(value=y_pred, num_or_size_splits=4, axis=1)
    
    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xc1 = K.minimum(x1, x1g)
    yc1 = K.minimum(y1, y1g)
    xc2 = K.maximum(x2, x2g)
    yc2 = K.maximum(y2, y2g)
    
    ###iou term###
    xA = K.maximum(x1g, x1)
    yA = K.maximum(y1g, y1)
    xB = K.minimum(x2g, x2)
    yB = K.minimum(y2g, y2)

    interArea = K.maximum(0.0, (xB - xA + 1)) * K.maximum(0.0, yB - yA + 1)
    boxAArea = (x2g - x1g + 1) * (y2g - y1g + 1)
    boxBArea = (x2 - x1 + 1) * (y2 - y1 + 1)

    iouk = interArea / (boxAArea + boxBArea - interArea + K.epsilon())
    ###
    
    ###distance term###
    c = K.square(xc2 - xc1) + K.square(yc2 - yc1) + K.epsilon()
    d = K.square(x_center - x_center_g) + K.square(y_center - y_center_g)
    u = d / c
    ###    
   
    # compute loss
    iou_loss = -tf.math.log(iouk)
    diou_loss = iou_loss + lamda_d * u
    
    if loss_type == "iou":
        return iou_loss
    elif loss_type == "diou":
        return diou_loss
    
   
def rpn_class_anchorless_loss_graph(rpn_match, rpn_class_logits):
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    indices = tf.where(K.not_equal(rpn_match, 0))
    target_class = tf.gather_nd(rpn_match, indices)
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)

    target_class = K.cast(K.equal(target_class, 1), tf.int32)
    
    
    pos_counts = K.sum(K.cast(K.equal(target_class, 1), tf.float32))        
    
    loss = binary_focal_loss(y_true = target_class, 
                             y_pred = rpn_class_logits[..., -1],
                             gamma=2, pos_weight=0.75, 
                             from_logits=True)
    
    loss = K.sum(loss)/pos_counts
    
    """

    loss = BinaryFocalLoss(gamma=2, pos_weight=0.75, from_logits=True)(
            y_true = target_class, 
            y_pred = rpn_class_logits[..., -1])
    
    loss = focal_loss(gamma=2.0, alpha=0.75, from_logits=True)(
                    y_true=target_class, 
                    y_pred=rpn_class_logits[..., -1])
    
    loss = K.sum(loss)/pos_counts
    """
    return loss




def rpn_class_loss_graph(rpn_match, rpn_class_logits):   
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    pos_indices = tf.where(K.equal(rpn_match, 1))
    neg_indices = tf.where(K.equal(rpn_match, -1))

    # Pick rows that contribute to the loss and filter out the rest.
    pos_rpn_class_logits = tf.gather_nd(rpn_class_logits, pos_indices)[..., -2:]
    pos_anchor_class = tf.gather_nd(anchor_class, pos_indices)
    neg_rpn_class_logits = tf.gather_nd(rpn_class_logits, neg_indices)[..., -2:]
    neg_anchor_class = tf.gather_nd(anchor_class, neg_indices)
    # Cross entropy loss    
    pos_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=pos_anchor_class, logits=pos_rpn_class_logits)
    neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=neg_anchor_class, logits=neg_rpn_class_logits)
    
    pos_loss = K.switch(tf.size(pos_loss) > 0, K.mean(pos_loss), tf.constant(0.0))
    neg_loss = K.switch(tf.size(neg_loss) > 0, K.mean(neg_loss), tf.constant(0.0))
    loss = 0.75*pos_loss + 0.25*neg_loss
    
    return loss


def rpn_bbox_anchorless_loss_graph(config, target_bbox, rpn_match, bbox_centers, rpn_bbox):  
    # Only positive anchors contribute to the loss
    rpn_match = tf.squeeze(rpn_match, -1)   
    indices = tf.where(K.equal(rpn_match, 1))
    
    # Pick bbox that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)[..., -4:]
    target_bbox = tf.gather_nd(target_bbox, indices)
    bbox_centers = tf.gather_nd(bbox_centers, indices)
    
    # convert boxes from [l,t,r,b] to [x1, y1, x2, y2]
    gx1 = bbox_centers[:, 0] - target_bbox[:, 1]
    gx2 = bbox_centers[:, 0] + target_bbox[:, 3]
    gy1 = bbox_centers[:, 1] - target_bbox[:, 0]
    gy2 = bbox_centers[:, 1] + target_bbox[:, 2]
    target_bbox = tf.stack([gx1, gy1, gx2, gy2], axis = 1)
    
    px1 = bbox_centers[:, 0] - rpn_bbox[:, 1]
    px2 = bbox_centers[:, 0] + rpn_bbox[:, 3]
    py1 = bbox_centers[:, 1] - rpn_bbox[:, 0]
    py2 = bbox_centers[:, 1] + rpn_bbox[:, 2]
    rpn_bbox = tf.stack([px1, py1, px2, py2], axis = 1)    
   
    # Compute CIoU loss
    ciou_loss = CIoULoss(target_bbox, rpn_bbox, 
                         config.BBOX_LOSS_LAMDA_D,
                         config.BBOX_LOSS_LAMDA_C)
    ciou_loss = K.switch(tf.size(ciou_loss) > 0, K.mean(ciou_loss), tf.constant(0.0))

    return ciou_loss




def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)[..., -4:]
    target_bbox = tf.gather_nd(target_bbox, indices)    
     
    loss = smooth_l1_loss(target_bbox, rpn_bbox, config.BBOX_SIGMA)
    
    loss = K.switch(tf.size(loss) > 0, K.mean(K.sum(loss, axis = 1)), tf.constant(0.0))
    
    return loss


def rpn_centerness_loss_graph(target_centerness, rpn_match, rpn_centerness_logits):
    # Only positive anchors contribute to the loss
    rpn_match = tf.squeeze(rpn_match, -1)
    target_centerness = K.squeeze(target_centerness, -1)
    
    indices = tf.where(K.equal(rpn_match, 1))
    
    # Pick bbox that contribute to the loss
    rpn_centerness_logits = tf.gather_nd(rpn_centerness_logits, indices)[..., -1]
    target_centerness = tf.gather_nd(target_centerness, indices)
    
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_centerness,
    #                                              logits=rpn_centerness_logits)
    loss = tf.keras.losses.MSE(target_centerness, tf.sigmoid(rpn_centerness_logits))
    #loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    
    return loss

  


def rpn_mask_loss_graph(config, target_masks, rpn_match, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    rpn_match = tf.squeeze(rpn_match, -1)
    pred_masks = tf.squeeze(pred_masks, -1)
    target_masks = tf.squeeze(target_masks, -1) 
    target_masks = tf.cast(target_masks, tf.float32)   
    
    # compute mask loss only for positive locations
    indices = tf.where(K.equal(rpn_match, 1))    
    target_masks = tf.gather_nd(target_masks, indices)
    pred_masks = tf.gather_nd(pred_masks, indices)   
      
    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(target_masks) > 0,
                    binary_logistic_regression_with_logits(target=target_masks, output=pred_masks),
                    tf.constant(0.0))
    loss = K.mean(loss)
    
    return loss

def rpn_mask_refine_loss_graph(config, target_masks, rpn_match, pred_masks):
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
    
    # compute mask loss only for positive locations
    indices = tf.where(K.equal(rpn_match, 1))    
    target_masks = tf.gather_nd(target_masks, indices)
    pred_masks = tf.gather_nd(pred_masks, indices)   
      
    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(target_masks) > 0,
                    binary_logistic_regression_with_logits(target=target_masks, output=pred_masks),
                    tf.constant(0.0))
    loss = K.mean(loss)
    
    return loss




def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

