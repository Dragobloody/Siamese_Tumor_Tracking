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
import keras.engine as KE

from siammask import utils, proposal

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_boxes, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.

    Returns: Target ROIs, bounding box shifts.    
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
   

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)      

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals[:, -4:], gt_boxes)   

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box
    negative_roi_bool = (roi_iou_max < 0.5)
    negative_indices = tf.where(negative_roi_bool)[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(tf.constant([1]), roi_gt_box_assignment)


    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois[:, -4:], roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV    

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N)], constant_values=-1)
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])

    return rois, roi_gt_class_ids, deltas


class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_boxes = inputs[1]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox"]
        outputs = utils.batch_slice(
            [proposals, gt_boxes],
            lambda x, y: detection_targets_graph(
                x, y, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 7),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas           
            ] 


############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rois, probs, deltas, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.       

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = proposal.apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = proposal.clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]        

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox],
            lambda x, y, z: refine_detections_graph(x, y, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)

    
def scale_penalty_graph(pred_bbox, kernel_bbox, penalty_k = 0.1):
    pred_bboxs_height = pred_bbox[:, :, 2] - pred_bbox[:, :, 0]
    pred_bboxs_width = pred_bbox[:, :, 3] - pred_bbox[:, :, 1]
    pred_bboxs_p = (pred_bboxs_height + pred_bboxs_width)/2
    # get prediction scales and ratios
    pred_bboxs_ratio = tf.math.divide(pred_bboxs_height,pred_bboxs_width)  
    pred_bboxs_scale = tf.math.sqrt(
                        tf.math.multiply((pred_bboxs_height+pred_bboxs_p),
                                         (pred_bboxs_width+pred_bboxs_p)))
    
    kernel_height = kernel_bbox[:, 2] - kernel_bbox[:, 0]
    kernel_width = kernel_bbox[:, 3] - kernel_bbox[:, 1]
    kernel_p = (kernel_height + kernel_width) / 2    
    # get kernel scale and ratio
    kernel_ratio = tf.math.divide(kernel_height, kernel_width)
    kernel_scale = tf.math.sqrt(
                    tf.math.multiply((kernel_height+kernel_p),
                                     (kernel_width+kernel_p)))
    kernel_ratio = tf.expand_dims(kernel_ratio, 1)
    kernel_scale = tf.expand_dims(kernel_scale, 1)

    
    max_ratios = tf.math.maximum(
                            tf.math.divide(pred_bboxs_ratio, kernel_ratio), 
                            tf.math.divide(kernel_ratio, pred_bboxs_ratio))
    max_scales = tf.math.maximum(
                            tf.math.divide(pred_bboxs_scale, kernel_scale), 
                            tf.math.divide(kernel_scale, pred_bboxs_scale))
    
    penalty = tf.math.exp(-penalty_k*(tf.math.multiply(max_ratios, max_scales)-1))
    penalty = tf.expand_dims(penalty, 2)
    
    return penalty


def postprocess_mask(masks, centers, config):
    pad_t = tf.cast(centers[..., 0] - config.KERNEL_IMAGE_SHAPE[0]//4, 'int32')
    pad_l = tf.cast(centers[..., 1] - config.KERNEL_IMAGE_SHAPE[1]//4, 'int32')
    pad_b = tf.cast(config.SEARCH_IMAGE_SHAPE[0] - (centers[..., 0] + config.KERNEL_IMAGE_SHAPE[0]//4), 'int32')
    pad_r = tf.cast(config.SEARCH_IMAGE_SHAPE[1] - (centers[..., 1] + config.KERNEL_IMAGE_SHAPE[1]//4), 'int32')
    
    def body(x):
        mask, t, l, b, r = x[0], x[1], x[2], x[3], x[4]
        full_mask = tf.pad(mask, [[t, b], [l, r], [0, 0]])
        
        return full_mask
    
    full_masks = tf.map_fn(body, [masks[0], pad_t[0], pad_l[0], pad_b[0], pad_r[0]], 
                           dtype = tf.float32, 
                           parallel_iterations=100)
    full_masks = tf.expand_dims(full_masks, 0)
    
    return full_masks



def postprocess_max_mask(masks, centers, config):
    k = 2
    if config.CROP:
        k = 4
    pad_t = tf.cast(centers[0] - config.KERNEL_IMAGE_SHAPE[0]//k, 'int32')
    pad_l = tf.cast(centers[1] - config.KERNEL_IMAGE_SHAPE[1]//k, 'int32')
    pad_b = tf.cast(config.SEARCH_IMAGE_SHAPE[0] - (centers[0] + config.KERNEL_IMAGE_SHAPE[0]//k), 'int32')
    pad_r = tf.cast(config.SEARCH_IMAGE_SHAPE[1] - (centers[1] + config.KERNEL_IMAGE_SHAPE[1]//k), 'int32')
    
    full_masks = tf.pad(masks, [[pad_t, pad_b], [pad_l, pad_r], [0, 0]])    
    
    return full_masks


class AnchorlessDetectionLayer(KE.Layer):
    def __init__(self, backbone_shapes, config=None, **kwargs):
        super(AnchorlessDetectionLayer, self).__init__(**kwargs)
        self.config = config
        self.backbone_shapes = backbone_shapes

    def call(self, inputs):
        pred_class = inputs[0]
        pred_centerness = inputs[1]
        pred_bbox = inputs[2]
        bbox_centers = inputs[3]
        input_image_metas = inputs[4]        
        
        # Multiply centerness with classification score        
        centerness = 1-pred_centerness[..., -1:]
        pred_class = tf.math.multiply(pred_class[..., -1:], 
                                      tf.math.exp(-np.power(centerness, 1.5)))
        pred_class = tf.concat([pred_centerness[..., :3], pred_class], axis = 2)
        
        pred_bbox = pred_bbox[..., -4:]
        if self.config.ANCHORLESS:
            # Convert bboxs from [l, t, r, b] to [x1, y1, x2, y2]            
            px1 = bbox_centers[:, :, 0] - pred_bbox[:, :, 1]
            px2 = bbox_centers[:, :, 0] + pred_bbox[:, :, 3]
            py1 = bbox_centers[:, :, 1] - pred_bbox[:, :, 0]
            py2 = bbox_centers[:, :, 1] + pred_bbox[:, :, 2]
            pred_bbox = tf.stack([px1, py1, px2, py2], axis = 2)
        else:
            # Convert bboxs from [dy, dx, log(dh), log(dw)] to [x1, y1, x2, y2]
            pyc = bbox_centers[:, :, 0] + pred_bbox[:, :, 0] * 64
            pxc = bbox_centers[:, :, 1] + pred_bbox[:, :, 1] * 64
            ph = 64 * tf.exp(pred_bbox[:, :, 2])
            pw = 64 * tf.exp(pred_bbox[:, :, 3])
            
            px1 = pyc - 0.5 * ph
            py1 = pxc - 0.5 * pw
            px2 = px1 + ph
            py2 = py1 + pw
            pred_bbox = tf.stack([px1, py1, px2, py2], axis = 2)
        
       
        # Ignore locations far away from the center and apply cosine window                 
        # define cosine window
        c = tf.signal.hamming_window(self.backbone_shapes[0], False)
        c = tf.tensordot(c, c, axes = 0)
        c = tf.reshape(c, [-1])
        cosine_window = c
        cosine_window = tf.expand_dims(cosine_window, 1)
        
        # get centered indices
        center = np.array(self.backbone_shapes)//2 + 1
        power = 1
        t = center[0] - center[0]//(2**power)
        b = center[0] + center[0]//(2**power)
        l = center[1] - center[1]//(2**(power+1))
        r = center[1] + center[1]//(2**(power+1))
        
        indices = tf.where(tf.math.logical_and(
                           tf.math.equal(pred_class[0, ..., 0], 0),  
                           tf.math.logical_and(
                                   tf.math.logical_and(
                                           tf.math.greater_equal(pred_class[0, ..., 1], t),
                                           tf.math.less_equal(pred_class[0, ..., 1], b)),
                                   tf.math.logical_and(
                                           tf.math.greater_equal(pred_class[0, ..., 2], l),
                                           tf.math.less_equal(pred_class[0, ..., 2], r)))
                                           ))
      
        # Apply cosine window
        pred_class = tf.math.multiply(pred_class[..., -1:], 
                                      cosine_window)
        pred_class = tf.concat([pred_centerness[..., :3], pred_class], axis = 2)
        
        # Get centered locations
        pred_class = tf.gather(pred_class, indices[:, 0], axis = 1)
        pred_bbox = tf.gather(pred_bbox, indices[:, 0], axis = 1)
       
        # Apply scale/ratio penalty
        kernel_bbox = input_image_metas[:, 11:]
        penalty = scale_penalty_graph(pred_bbox, kernel_bbox, penalty_k = 0.3)
        pred_class = tf.concat([pred_class[..., :-1],
                                tf.math.multiply(pred_class[..., -1:],
                                                 penalty)], axis = 2)
                    
        # Get max predictions
        max_indices = tf.math.argmax(pred_class[..., -1], axis = 1, output_type='int32')
        batch_idx = tf.range(tf.shape(max_indices)[0])
        max_indices = tf.stack([batch_idx, max_indices], axis = 1)

        max_pred_class = tf.gather_nd(pred_class, max_indices)
        max_pred_bbox = tf.gather_nd(pred_bbox, max_indices)        
        max_detections = tf.concat([max_pred_class, max_pred_bbox], axis = 1)        
        
        return [pred_class, pred_bbox, max_detections, penalty]
    
    def compute_output_shape(self, input_shape):
         return [           
            (None, None, 4),           
            (None, None, 4),  
            (None, 8),        
            (None, None, 1)
            ] 
    

class AnchorlessDetectionLayerRefine(KE.Layer):
    def __init__(self, backbone_shapes, config=None, **kwargs):
        super(AnchorlessDetectionLayerRefine, self).__init__(**kwargs)
        self.config = config
        self.backbone_shapes = backbone_shapes

    def call(self, inputs):
        pred_class = inputs[0]
        pred_centerness = inputs[1]
        pred_bbox = inputs[2]
        bbox_centers = inputs[3]
        input_image_metas = inputs[4]  
        feature_maps = inputs[5]
     
        
        # reshape feature maps
        s = tf.shape(feature_maps)
        feature_maps = tf.reshape(feature_maps, (s[0], -1, s[-1]))
        
        # Multiply centerness with classification score
        centerness = 1-pred_centerness[..., -1:]
        pred_class = tf.math.multiply(pred_class[..., -1:], 
                                      tf.math.exp(-np.power(centerness, 1.5)))
        pred_class = tf.concat([pred_centerness[..., :3], pred_class], axis = 2)
        
        pred_bbox = pred_bbox[..., -4:]
        if self.config.ANCHORLESS:
            # Convert bboxs from [l, t, r, b] to [x1, y1, x2, y2]            
            px1 = bbox_centers[:, :, 0] - pred_bbox[:, :, 1]
            px2 = bbox_centers[:, :, 0] + pred_bbox[:, :, 3]
            py1 = bbox_centers[:, :, 1] - pred_bbox[:, :, 0]
            py2 = bbox_centers[:, :, 1] + pred_bbox[:, :, 2]
            pred_bbox = tf.stack([px1, py1, px2, py2], axis = 2)
        else:
            # Convert bboxs from [dy, dx, log(dh), log(dw)] to [x1, y1, x2, y2]
            pyc = bbox_centers[:, :, 0] + pred_bbox[:, :, 0] * 64
            pxc = bbox_centers[:, :, 1] + pred_bbox[:, :, 1] * 64
            ph = 64 * tf.exp(pred_bbox[:, :, 2])
            pw = 64 * tf.exp(pred_bbox[:, :, 3])
            
            px1 = pyc - 0.5 * ph
            py1 = pxc - 0.5 * pw
            px2 = px1 + ph
            py2 = py1 + pw
            pred_bbox = tf.stack([px1, py1, px2, py2], axis = 2)
        
       
        # Ignore locations far away from the center and apply cosine window                 
        # define cosine window
        c = tf.signal.hamming_window(self.backbone_shapes[0], False)
        c = tf.tensordot(c, c, axes = 0)
        c = tf.reshape(c, [-1])
        cosine_window = c
        cosine_window = tf.expand_dims(cosine_window, 1)
        
        # get centered indices
        center = np.array(self.backbone_shapes)//2 + 1
        power = 1
        t = center[0] - center[0]//(2**power)
        b = center[0] + center[0]//(2**power)
        l = center[1] - center[1]//(2**(power+1))
        r = center[1] + center[1]//(2**(power+1))
        
        indices = tf.where(tf.math.logical_and(
                           tf.math.equal(pred_class[0, ..., 0], 0),  
                           tf.math.logical_and(
                                   tf.math.logical_and(
                                           tf.math.greater_equal(pred_class[0, ..., 1], t),
                                           tf.math.less_equal(pred_class[0, ..., 1], b)),
                                   tf.math.logical_and(
                                           tf.math.greater_equal(pred_class[0, ..., 2], l),
                                           tf.math.less_equal(pred_class[0, ..., 2], r)))
                                           ))
      
        # Apply cosine window
        pred_class = tf.math.multiply(pred_class[..., -1:], 
                                      cosine_window)
        pred_class = tf.concat([pred_centerness[..., :3], pred_class], axis = 2)
        
        # Get centered locations
        pred_class = tf.gather(pred_class, indices[:, 0], axis = 1)
        pred_bbox = tf.gather(pred_bbox, indices[:, 0], axis = 1)
        bbox_centers = tf.gather(bbox_centers, indices[:, 0], axis = 1)
        feature_maps = tf.gather(feature_maps, indices[:, 0], axis = 1)

        # Apply scale/ratio penalty
        kernel_bbox = input_image_metas[:, 11:]
        penalty = scale_penalty_graph(pred_bbox, kernel_bbox, penalty_k = 0.1)
        pred_class = tf.concat([pred_class[..., :-1],
                                tf.math.multiply(pred_class[..., -1:],
                                                 penalty)], axis = 2)
                    
        # Get max predictions
        max_indices = tf.math.argmax(pred_class[..., -1], axis = 1, output_type='int32')
        batch_idx = tf.range(tf.shape(max_indices)[0])
        max_indices = tf.stack([batch_idx, max_indices], axis = 1)

        max_pred_class = tf.gather_nd(pred_class, max_indices)
        max_pred_bbox = tf.gather_nd(pred_bbox, max_indices)        
        max_detections = tf.concat([max_pred_class, max_pred_bbox], axis = 1)        
        max_bbox_centers = tf.gather_nd(bbox_centers, max_indices)
        mask_feature_maps = tf.gather_nd(feature_maps, max_indices)
        mask_feature_maps = tf.reshape(mask_feature_maps, (s[0], 1, 1, 1, s[-1]))
              
        
        return [pred_class, pred_bbox, max_detections, penalty, mask_feature_maps, max_bbox_centers]
    
    def compute_output_shape(self, input_shape):
         return [           
            (None, None, 4),           
            (None, None, 4), 
            (None, 8),  
            (None, None, 1),
            (None, None, 1, 1, self.config.TOP_DOWN_PYRAMID_SIZE),
            (None, 2),  
            ] 

class AnchorlessDetectionLayerMask(KE.Layer):
    def __init__(self, backbone_shapes, config=None, **kwargs):
        super(AnchorlessDetectionLayerMask, self).__init__(**kwargs)
        self.config = config
        self.backbone_shapes = backbone_shapes

    def call(self, inputs):
        pred_class = inputs[0]
        pred_centerness = inputs[1]
        pred_bbox = inputs[2]
        pred_masks = inputs[3]
        bbox_centers = inputs[4]
        input_image_metas = inputs[5]        
        
        # Multiply centerness with classification score
        centerness = 1-pred_centerness[..., -1:]
        pred_class = tf.math.multiply(pred_class[..., -1:], 
                                      tf.math.exp(-np.power(centerness, 1.5)))
        pred_class = tf.concat([pred_centerness[..., :3], pred_class], axis = 2)
        
        pred_bbox = pred_bbox[..., -4:]
        if self.config.ANCHORLESS:
            # Convert bboxs from [l, t, r, b] to [x1, y1, x2, y2]            
            px1 = bbox_centers[:, :, 0] - pred_bbox[:, :, 1]
            px2 = bbox_centers[:, :, 0] + pred_bbox[:, :, 3]
            py1 = bbox_centers[:, :, 1] - pred_bbox[:, :, 0]
            py2 = bbox_centers[:, :, 1] + pred_bbox[:, :, 2]
            pred_bbox = tf.stack([px1, py1, px2, py2], axis = 2)
        else:
            # Convert bboxs from [dy, dx, log(dh), log(dw)] to [x1, y1, x2, y2]
            pyc = bbox_centers[:, :, 0] + pred_bbox[:, :, 0] * 64
            pxc = bbox_centers[:, :, 1] + pred_bbox[:, :, 1] * 64
            ph = 64 * tf.exp(pred_bbox[:, :, 2])
            pw = 64 * tf.exp(pred_bbox[:, :, 3])
            
            px1 = pyc - 0.5 * ph
            py1 = pxc - 0.5 * pw
            px2 = px1 + ph
            py2 = py1 + pw
            pred_bbox = tf.stack([px1, py1, px2, py2], axis = 2)
        
       
        # Ignore locations far away from the center and apply cosine window                 
        # define cosine window
        c = tf.signal.hamming_window(self.backbone_shapes[0], False)
        c = tf.tensordot(c, c, axes = 0)
        c = tf.reshape(c, [-1])
        cosine_window = c
        cosine_window = tf.expand_dims(cosine_window, 1)
        
        # get centered indices
        center = np.array(self.backbone_shapes)//2 + 1
        power = 1
        t = center[0] - center[0]//(2**power)
        b = center[0] + center[0]//(2**power)
        l = center[1] - center[1]//(2**(power+1))
        r = center[1] + center[1]//(2**(power+1))
        
        indices = tf.where(tf.math.logical_and(
                           tf.math.equal(pred_class[0, ..., 0], 0),  
                           tf.math.logical_and(
                                   tf.math.logical_and(
                                           tf.math.greater_equal(pred_class[0, ..., 1], t),
                                           tf.math.less_equal(pred_class[0, ..., 1], b)),
                                   tf.math.logical_and(
                                           tf.math.greater_equal(pred_class[0, ..., 2], l),
                                           tf.math.less_equal(pred_class[0, ..., 2], r)))
                                           ))
      
        # Apply cosine window
        pred_class = tf.math.multiply(pred_class[..., -1:], 
                                      cosine_window)
        pred_class = tf.concat([pred_centerness[..., :3], pred_class], axis = 2)
        
        # Get centered locations
        pred_class = tf.gather(pred_class, indices[:, 0], axis = 1)
        pred_bbox = tf.gather(pred_bbox, indices[:, 0], axis = 1)
        if self.config.MASK:
            bbox_centers = tf.gather(bbox_centers, indices[:, 0], axis = 1)
            pred_masks = tf.gather(pred_masks, indices[:, 0], axis = 1)
            
        # Apply scale/ratio penalty
        kernel_bbox = input_image_metas[:, 11:]
        penalty = scale_penalty_graph(pred_bbox, kernel_bbox, penalty_k = 0.1)
        pred_class = tf.concat([pred_class[..., :-1],
                                tf.math.multiply(pred_class[..., -1:],
                                                 penalty)], axis = 2)

                    
        # Get max predictions
        max_indices = tf.math.argmax(pred_class[..., -1], axis = 1, output_type='int32')
        batch_idx = tf.range(tf.shape(max_indices)[0])
        max_indices = tf.stack([batch_idx, max_indices], axis = 1)

        max_pred_class = tf.gather_nd(pred_class, max_indices)
        max_pred_bbox = tf.gather_nd(pred_bbox, max_indices)
        if self.config.MASK:
            max_bbox_centers = tf.gather_nd(bbox_centers, max_indices)
            max_pred_mask = tf.gather_nd(pred_masks, max_indices)
            max_pred_mask = utils.batch_slice(
                    [max_pred_mask, max_bbox_centers], 
                    lambda x, y: postprocess_max_mask(x, y, self.config),
                    self.config.IMAGES_PER_GPU)
            
        max_detections = tf.concat([max_pred_class, max_pred_bbox], axis = 1)        
        
        return [pred_class, pred_bbox, pred_masks, max_detections, max_pred_mask, penalty]
    
    def compute_output_shape(self, input_shape):
         return [           
            (None, None, 4),           
            (None, None, 4),  
            (None, None, None, None, 1), 
            (None, 8),        
            (None, None, None, 1), 
            (None, None, 1)
            ] 
    
    
    
    

    
    
def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """ 
    
    original_image_shape = meta[:, 0:3]
    image_shape = meta[:, 3:6]
    window = meta[:, 6:10]  # (y1, x1, y2, x2) window of image in in pixels    
    active_class_ids = meta[:, 10:]
    return {       
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,       
        "active_class_ids": active_class_ids,
    }
    
    
def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)
    
    
