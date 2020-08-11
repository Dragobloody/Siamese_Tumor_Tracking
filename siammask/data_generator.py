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

from siammask import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

############################################################
#  Data Formatting
############################################################

def compose_image_meta(original_search_image_shape, original_search_area_shape, 
                       search_area_shape, center, kernel_bbox):
    """Takes attributes of an image and puts them in one 1D array.
   
    original_search_image_shape: [H, W, C] before cropping(usually 384x512)
    original_search_area_shape: [H, W, C] after cropping and before resizing
    search_area_shape: [H, W, C] after cropping and resizing(usually 255x255)
    center: [x, y] coordinates of the kernel/search image area
    
    """
    meta = np.array(
        list(original_search_image_shape) +  # size=3   
        list(original_search_area_shape) +   # size=3
        list(search_area_shape) +            # size=3
        list(center) +                       # size=2 (x, y) in image coordinates   
        list(kernel_bbox)                    # size=4 (x1, y1, x2, y2) 
        
    )
    return meta


############################################################
#  Data Generator
############################################################

def load_image_gt(config, kernel_image, search_image, kernel_mask=None, search_mask=None, kernel_bbox=None, mode = 'training'):
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
    if len(kernel_image.shape) < 3:
        kernel_image = kernel_image[..., np.newaxis]
    if len(search_image.shape) < 3:
        search_image = search_image[..., np.newaxis]     
    
    # Flip image in training mode
    if mode == 'training':
        flip = random.randint(0, 1)
        if flip == 1:
            kernel_image = np.flip(kernel_image, axis=1)
            search_image = np.flip(search_image, axis=1)
            kernel_mask = np.flip(kernel_mask, axis=1)
            search_mask = np.flip(search_mask, axis=1)        
            
        
        # random contrast transform
        gamma = random.uniform(0.3, 2)
        kernel_image = utils.gamma_transform(kernel_image, gamma)
        search_image = utils.gamma_transform(search_image, gamma) 
        
                
        
    # zero mean unit variance
    #kernel_image = utils.normalize(kernel_image)
    #search_image = utils.normalize(search_image)
    
   
    
       
    original_shape = search_image.shape
    # get target center
    if kernel_bbox is None:
        if len(kernel_mask.shape) < 3:
            kernel_mask = kernel_mask[..., np.newaxis] 
        kernel_bbox = utils.extract_bboxes(kernel_mask)
        
    kernel_center = [(kernel_bbox[0, 0] + kernel_bbox[0, 2])//2, (kernel_bbox[0, 1] + kernel_bbox[0, 3])//2]    
    search_center = kernel_center.copy()
    kernel_height = kernel_bbox[0, 2] - kernel_bbox[0, 0] + 1
    kernel_width = kernel_bbox[0, 3] - kernel_bbox[0, 1] + 1
    p = (kernel_height + kernel_width) // 2
    
    #p = p * 1.5
    if config.CROP:
        p = p * 1.5
        
    kernel_size = np.round(np.sqrt((kernel_height + p)*(kernel_width + p))).astype('int16')
    
    shift_x = shift_y = 0
    if mode == 'training':   
                
        if random.randint(0, 3) == -1:
            if random.randint(0, 1) == 1:
                shift_x = ((random.randint(0, 1)*2 - 1))*random.randint(32, 64)
                shift_y = random.randint(-64, 64)
            else:
                shift_y = ((random.randint(0, 1)*2 - 1))*random.randint(32, 64)
                shift_x = random.randint(-64, 64)
            
            #kernel_image = np.random.random(kernel_image.shape)
            #kernel_image = utils.normalize(kernel_image)
            
        else:
                
            shift_x = random.randint(-8, 8)
            shift_y = random.randint(-8, 8)     

        kernel_center[0] += shift_x
        kernel_center[1] += shift_y 
        
        #search_center[0] += random.randint(32, 33)
        #search_center[1] += random.randint(32, 33)
        
        if random.randint(0, 1) == 1:
            kernel_rescale = 2**((random.randint(0, 1)*2 - 1)/random.randint(6, 10))
            kernel_size = np.round(kernel_size*kernel_rescale).astype('int16')
    else:
        if search_mask is not None:
            #kernel_center[0] += random.randint(16, 32)
            #kernel_center[1] += random.randint(16, 32)
            search_center[0] += random.randint(-16, 16)
            search_center[1] += random.randint(-16, 16)
        else:
            #kernel_center[0] += random.randint(8, 16)
            #kernel_center[1] += random.randint(8, 16)
            kernel_size = kernel_size

    #kernel_image = np.zeros(kernel_image.shape, dtype='float32')
    #kernel_image = np.random.random(kernel_image.shape)
    #kernel_image = utils.normalize(kernel_image)
    
    original_bbox = None
    search_bbox = None
    if search_mask is not None:
        if len(search_mask.shape) < 3:
            search_mask = search_mask[..., np.newaxis]       
        # get original gt bbox
        original_bbox = utils.extract_bboxes(search_mask)    
        
        if mode == 'training':
            search_center = [(original_bbox[0, 0] + original_bbox[0, 2])//2, (original_bbox[0, 1] + original_bbox[0, 3])//2]    
            search_center[0] += random.randint(-32, 32)
            search_center[1] += random.randint(-32, 32) 
        
        
        search_mask = utils.crop_image(search_mask, search_center, kernel_size*2, kernel_size*2)
        search_mask = scipy.ndimage.zoom(search_mask[..., 0], 
                                         zoom = config.SEARCH_IMAGE_SHAPE[0]/search_mask.shape[0],
                                         order=0,
                                         )[..., np.newaxis]
        # get search gt bbox
        search_bbox = utils.extract_bboxes(search_mask)  
        
    
    # get kernel and search regions
    if kernel_mask is not None:
        kernel_mask = utils.crop_image(kernel_mask, kernel_center, kernel_size, kernel_size)
        kernel_mask = scipy.ndimage.zoom(kernel_mask[..., 0], 
                                         zoom = config.KERNEL_IMAGE_SHAPE[0]/kernel_mask.shape[0],
                                         order=0,
                                         )[..., np.newaxis]
        
    
    kernel_image = utils.crop_image(kernel_image, kernel_center, kernel_size, kernel_size)
    search_image = utils.crop_image(search_image, search_center, kernel_size*2, kernel_size*2)
    search_original_shape = search_image.shape  
    #print(search_original_shape)
    
    # resize data to 127x127 and 255x255    
    kernel_image = scipy.ndimage.zoom(kernel_image[..., 0], 
                                      zoom = config.KERNEL_IMAGE_SHAPE[0]/kernel_image.shape[0],
                                      order=0,
                                      )[..., np.newaxis]
    
    search_image = scipy.ndimage.zoom(search_image[..., 0], 
                                        zoom = config.SEARCH_IMAGE_SHAPE[0]/search_image.shape[0],
                                        order=0,
                                        )[..., np.newaxis]

    #print(search_image.shape)
    if config.IMAGE_CHANNEL_COUNT == 3:
        kernel_image = kernel_image * np.ones(3, dtype=int)[None, None, :]  
        search_image = search_image * np.ones(3, dtype=int)[None, None, :] 
    
    # Image meta data    
    search_image_meta = compose_image_meta(original_shape, search_original_shape,
                                    search_image.shape, search_center, kernel_bbox[0])

    return kernel_image, search_image, search_image_meta, kernel_mask, search_mask, search_bbox, original_bbox, shift_x, shift_y


def build_anchorless_rpn_targets(mappings, gt_boxes, config, 
                                 shift_x=0, shift_y=0, bbox_ratio=1.0):
    #compute anchors
    anchors = np.zeros((mappings.shape[0], 4), dtype=np.int32)
    anchors[:, 0] = mappings[:, 0] - 32
    anchors[:, 1] = mappings[:, 1] - 32
    anchors[:, 2] = mappings[:, 0] + 32
    anchors[:, 3] = mappings[:, 1] + 32

    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([mappings.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((mappings.shape[0], 4), dtype=np.float32)
    rpn_bbox_anchor = np.zeros((mappings.shape[0], 4), dtype=np.float32)

    rpn_centerness = np.zeros((mappings.shape[0], 1), dtype=np.float32)
    
    gt_boxes_height = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_boxes_width = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_center_y = gt_boxes[:, 0] + 0.5 * gt_boxes_height
    gt_center_x = gt_boxes[:, 1] + 0.5 * gt_boxes_width
    
    l = (mappings[:, 1] - gt_boxes[:, 1]).astype('float32')
    r = (gt_boxes[:, 3]  - mappings[:, 1]).astype('float32')
    t = (mappings[:, 0] - gt_boxes[:, 0]).astype('float32')
    b = (gt_boxes[:, 2] - mappings[:, 0]).astype('float32')
    boxes = np.stack([l,t,r,b], axis = 1)    
    
    poz_indices = (l>gt_boxes_width*(1-bbox_ratio))&(r>gt_boxes_width*(1-bbox_ratio))&(t>gt_boxes_height*(1-bbox_ratio))&(b>gt_boxes_height*(1-bbox_ratio))
    neg_indices = (l<=0) | (r<=0) | (t<=0) | (b<=0)
    neutral_poz_indices = (l>0) & (r>0) & (t>0) & (b>0)

    #poz_boxes = boxes[poz_indices]
    neutral_poz_boxes = boxes[neutral_poz_indices]
    rpn_match[poz_indices] = 1
    rpn_match[neg_indices] = -1
    
    rpn_bbox[neutral_poz_indices] = neutral_poz_boxes
    
    
    if abs(shift_x) > 16 or abs(shift_y) > 16:
        rpn_match[:] = -1
    
    
    
    # Compute the bbox refinement that the RPN should predict.   
    boxes_mrcnn = np.stack([
        (gt_center_y - mappings[:, 0]) / 64,
        (gt_center_x - mappings[:, 1]) / 64,
        np.log(gt_boxes_height / np.broadcast_to([64], (mappings.shape[0]))),
        np.log(gt_boxes_width / np.broadcast_to([64], (mappings.shape[0])))], axis=1)
    rpn_bbox_anchor[neutral_poz_indices] = boxes_mrcnn[neutral_poz_indices]
    
    
    # compute target centerness
    min_lr = np.minimum(neutral_poz_boxes[:, 0], neutral_poz_boxes[:, 2])
    min_tb = np.minimum(neutral_poz_boxes[:, 1], neutral_poz_boxes[:, 3])
    max_lr = np.maximum(neutral_poz_boxes[:, 0], neutral_poz_boxes[:, 2])
    max_tb = np.maximum(neutral_poz_boxes[:, 1], neutral_poz_boxes[:, 3])

    centerness = np.sqrt(min_lr/max_lr * min_tb/max_tb)
    rpn_centerness[neutral_poz_indices] = centerness[:, np.newaxis]
    
    return rpn_match, rpn_bbox, rpn_centerness, rpn_bbox_anchor
   


def build_rpn_targets(image_shape, anchors, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]   
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    
    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 4)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]       
        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


