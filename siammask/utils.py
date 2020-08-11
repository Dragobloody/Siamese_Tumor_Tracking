"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import logging
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)




def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def crop_image(image, center, width, height):
    image_shape = image.shape
    x1 = max(0, center[0]-height//2)
    x2 = min(center[0]+height//2, image_shape[0])
    y1 = max(0, center[1]-width//2)
    y2 = min(center[1]+width//2, image_shape[1])
    
    pad_t = max(0, height//2 - center[0])
    pad_b = max(0, height//2 + center[0] - image_shape[0])
    pad_l = max(0, width//2 - center[1])
    pad_r = max(0, width//2 + center[1] - image_shape[1])
    
    image = np.pad(image[x1:x2, y1:y2], 
                   ((pad_t, pad_b), 
                    (pad_l, pad_r), 
                    (0, 0)))
    
    return image


def resize_image(image):
    """Resizes an image keeping the aspect ratio unchanged.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.   
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """   
    
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)    
    padding = [(0, 0), (0, 0), (0, 0)]
    
    # Height
    if h % 64 > 0:
        max_h = h - (h % 64) + 64
        top_pad = (max_h - h) // 2
        bottom_pad = max_h - h - top_pad
    else:
        top_pad = bottom_pad = 0
    # Width
    if w % 64 > 0:
        max_w = w - (w % 64) + 64
        left_pad = (max_w - w) // 2
        right_pad = max_w - w - left_pad
    else:
        left_pad = right_pad = 0
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=np.min(image))
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    
    return image.astype(image_dtype), window, padding


def resize_mask(mask, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.
    
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """      
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m, (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass



def fit_mask_to_search_area(mask, pos, image_shape = (256, 256), stride = 4):
    mask_shape = mask.shape
    final_mask = np.zeros(image_shape, dtype = mask.dtype)
    
    shift_x = mask_shape[0]//2 + mask_shape[0]%2
    shift_y = mask_shape[1]//2 + mask_shape[1]%2
    x_center = mask_shape[0]//2 + pos[0]*stride
    y_center =  mask_shape[1]//2 + pos[1]*stride
    
    final_mask[x_center-mask_shape[0]//2:x_center+shift_x,
               y_center-mask_shape[1]//2:y_center+shift_y] = mask          
   
    
    return final_mask


def unmold_anchorless_detections(detections, image_metas):
    zoom = image_metas[:, 3]/image_metas[:, 6]
    detections[:, -4:] = detections[:, -4:]*zoom[..., np.newaxis]   
    
    shift_h = image_metas[:, 9] - image_metas[:, 3]//2
    shift_w = image_metas[:, 10] - image_metas[:, 4]//2
    shift = [shift_h, shift_w, shift_h, shift_w]
    shift = np.stack(shift, axis = 1)
    
    detections[:, -4:] = detections[:, -4:] + shift
    detections[:, -4:] = np.round(detections[:, -4:])
    
    return detections

def unmold_anchorless_masks(config, detections, masks, image_metas):
    unmolded_masks = []    
    #stride_idx = detections[:, 0].astype('int32')
    #locations = detections[:, 1:3].astype('int32')
    mask_centers = image_metas[:, 9:11].astype('int32')
    mask_zooms = image_metas[:, 3]/image_metas[:, 6]

    for i in range(masks.shape[0]):
        full_mask = np.zeros(image_metas[i, 0:2].astype('int32'), masks.dtype)
        """
        final_mask = fit_mask_to_search_area(masks[i], 
                      locations[i], 
                      image_shape = config.SEARCH_IMAGE_SHAPE[:2], 
                      stride = config.BACKBONE_STRIDES[stride_idx[i]])
        """
        final_mask = masks[i]
        final_mask = scipy.ndimage.zoom(final_mask, mask_zooms[i], order=0)
        shift_x = final_mask.shape[0]//2 + final_mask.shape[0]%2
        shift_y = final_mask.shape[1]//2 + final_mask.shape[1]%2
    
        x1 = max(0, mask_centers[i, 0]-final_mask.shape[0]//2)
        x2 = min(full_mask.shape[0], mask_centers[i, 0]+shift_x)
        y1 = max(0, mask_centers[i, 1]-final_mask.shape[1]//2)
        y2 = min(full_mask.shape[1], mask_centers[i, 1]+shift_y)
        
        x1_s = max(0, final_mask.shape[0]//2-mask_centers[i, 0])
        y1_s = max(0, final_mask.shape[1]//2-mask_centers[i, 1])
        
        full_mask[x1:x2, y1:y2] = final_mask[x1_s:x1_s+x2-x1, y1_s:y1_s+y2-y1] 
                
        unmolded_masks.append(full_mask)
    
    unmolded_masks = np.stack(unmolded_masks, axis = 0)
    
    return unmolded_masks

def unmold_detections(detections, image_meta):
    detections = denorm_boxes(detections, image_meta[6:8])
    zoom = image_meta[3]/image_meta[6]
    detections = detections*zoom    
    
    shift_h = image_meta[9] - image_meta[3]//2
    shift_w = image_meta[10] - image_meta[4]//2
    shift = [shift_h, shift_w, shift_h, shift_w]
    
    detections = detections + shift
    
    return np.round(detections).astype('int32')
    


def unmold_mask(mask, pos, image_meta, config):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 127x127 mask.  

    Returns a binary mask with the same size as the original image.
    """
    full_mask = np.zeros(image_meta[0:2], mask.dtype)
    center_x, center_y = image_meta[9], image_meta[10]
    mask = fit_mask_to_search_area(mask, pos, image_meta[6:8], config.BACKBONE_STRIDES[0])
   
    mask = scipy.ndimage.interpolation.zoom(mask, 
                                            zoom = image_meta[3]/image_meta[6])
    
    shift_x = mask.shape[0]//2 + mask.shape[0]%2
    shift_y = mask.shape[1]//2 + mask.shape[1]%2
    
    
    full_mask[center_x-mask.shape[0]//2:center_x+shift_x,
               center_y-mask.shape[1]//2:center_y+shift_y] = mask 
    
              
    full_mask = np.round(full_mask).astype('uint8')    
   
    return full_mask


def scale_penalty(rois, kernel_bbox, penalty_k = 0.1):
    rois_height = rois[:, 2] - rois[:, 0]
    rois_width = rois[:, 3] - rois[:, 1]
    rois_p = (rois_height + rois_width)/2
    # get prediction scales and ratios
    rois_ratio = rois_height/rois_width  
    rois_scale = np.sqrt((rois_height+rois_p)*(rois_width+rois_p))
    
    kernel_height = kernel_bbox[2] - kernel_bbox[0]
    kernel_width = kernel_bbox[3] - kernel_bbox[1]
    kernel_p = (kernel_height + kernel_width) / 2    
    # get kernel scale and ratio
    kernel_ratio = kernel_height/kernel_width
    kernel_scale = np.sqrt((kernel_height+kernel_p)*(kernel_width+kernel_p))
    
    max_ratios = np.maximum(rois_ratio/kernel_ratio, kernel_ratio/rois_ratio)
    max_scales = np.maximum(rois_scale/kernel_scale, kernel_scale/rois_scale)
    
    penalty = np.exp(- penalty_k * (max_ratios*max_scales-1))
    
    return penalty


def top_k_predictions(config, scores, rois, masks, image_meta, k = 3):    
    top_k_results = {}
    # apply scale penalty
    kernel_bbox = image_meta[11:]
    penalty = scale_penalty(rois, kernel_bbox)
    scores = scores*penalty
    
    # sort predictions
    top_k_idx = np.argsort(scores)[::-1]
    top_k_scores = scores[top_k_idx]
    top_k_rois = rois[top_k_idx]
    
    # apply non-max supression
    nms_idx = non_max_suppression(top_k_rois, top_k_scores, 0.7)
    # get top k predictions
    top_k_idx = top_k_idx[nms_idx[:k]]
    top_k_scores = top_k_scores[nms_idx[:k]]
    top_k_rois = top_k_rois[nms_idx[:k]]
    
    if masks is not None:
        top_k_pos = np.column_stack(np.unravel_index(top_k_idx, (config.FEATURES_SIZE[0], 
                                                                 config.FEATURES_SIZE[0], 
                                                                 len(config.RPN_ANCHOR_RATIOS))))
        top_k_masks = masks[top_k_pos[:, 0], top_k_pos[:, 1], ...] 
        
        # unmold masks   
        top_k_full_masks = []
        for i in range(k):
            full_mask = unmold_mask(top_k_masks[i], top_k_pos[i, :2], image_meta, config)
            top_k_full_masks.append(full_mask)
        top_k_full_masks = np.stack(top_k_full_masks, axis = 0)
        
        top_k_mask_rois = extract_bboxes(np.transpose(top_k_full_masks, (1, 2, 0)))
    
        top_k_results['masks'] = top_k_masks
        top_k_results['full_masks'] = top_k_full_masks
        top_k_results['mask_rois'] = top_k_mask_rois
        
    top_k_results['scores'] = top_k_scores
    top_k_results['rois'] = top_k_rois
    
    
    
    return top_k_results
    

############################################################
#  Anchors
############################################################


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # only for siammask
    shape = shape//2 + 1 
    #shape = shape - 1
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride + 64
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride + 64
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))        
    return np.concatenate(anchors, axis=0).astype('float32')


def generate_mappings(shape, feature_stride, mapping_stride, config):
    shape = shape//2 + 1
    s = 64
    if config.CROP:
        shape = config.FEATURES_SHAPE
        s = 32
    # Enumerate shifts in feature space
    box_centers_y = np.arange(0, shape[0], mapping_stride) * feature_stride + s
    box_centers_x = np.arange(0, shape[1], mapping_stride) * feature_stride + s
    box_centers_x, box_centers_y = np.meshgrid(box_centers_x, box_centers_y)
    box_centers_x = box_centers_x.flatten()
    box_centers_y = box_centers_y.flatten()
    
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=1)
    
    return box_centers

def generate_mappings_mrcnn(shape, feature_stride, mapping_stride):
     # Enumerate shifts in feature space
    box_centers_y = np.arange(0, shape[0], mapping_stride) * feature_stride + feature_stride//2
    box_centers_x = np.arange(0, shape[1], mapping_stride) * feature_stride + feature_stride//2
    box_centers_x, box_centers_y = np.meshgrid(box_centers_x, box_centers_y)
    box_centers_x = box_centers_x.flatten()
    box_centers_y = box_centers_y.flatten()
    
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=1)
    
    return box_centers

def generate_pyramid_mappings(feature_shapes, feature_strides,
                             mapping_stride, config):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Mappings
    # [mapping_count, (y, x)]
    mappings = []
    for i in range(len(feature_shapes)):
        mappings.append(generate_mappings(feature_shapes[i],
                                        feature_strides[i], mapping_stride, config))        
    return np.concatenate(mappings, axis=0).astype('float32')

def generate_pyramid_mappings_mrcnn(feature_shapes, feature_strides,
                             mapping_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Mappings
    # [mapping_count, (y, x)]
    mappings = []
    for i in range(len(feature_shapes)):
        mappings.append(generate_mappings_mrcnn(feature_shapes[i],
                                        feature_strides[i], mapping_stride))        
    return np.concatenate(mappings, axis=0).astype('float32')



############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    
    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps =\
            compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)
        
        
        
def bboxs_to_masks(bboxs, image_shape = (384, 512)):
    masks = np.zeros(bboxs.shape[:-1] + image_shape, dtype = 'uint8')
    initial_shape = masks.shape
    masks = np.reshape(masks, (-1, image_shape[0], image_shape[1]))
    bboxs = np.reshape(bboxs, (-1, 4))
    for i in range(masks.shape[0]):
        masks[i, bboxs[i, 0]:bboxs[i, 2], bboxs[i, 1]:bboxs[i, 3]] = 1
    
    return np.reshape(masks, initial_shape)


def incorporate_location_graph(config, features, level, shape):    
    shape = shape//2 +1 
    idx_0, idx_1, idx_2 = np.meshgrid(np.arange(0, shape[0], 1), np.arange(0, shape[1], 1), np.ones(len(config.RPN_ANCHOR_RATIOS), dtype = 'int32')*level)
    idx_0 = idx_0.flatten()
    idx_1 = idx_1.flatten()
    idx_2 = idx_2.flatten()
    idx = np.stack([idx_2, idx_0, idx_1], axis = 1)
    idx = np.broadcast_to(idx, (config.BATCH_SIZE,) + idx.shape)
    idx = tf.constant(idx, dtype = tf.float32)
    features = tf.concat([idx, features], axis = 2)
    
    return features
    
    
def log_transform(images): 
    assert len(images.shape) == 3, "Images format is not correct"
    
    c = 1/np.log(1 + np.max(images, axis = (1,2)))
    images = np.log(images+1)
    images = np.multiply(c[..., np.newaxis, np.newaxis], images)
    images = images*2 - 1
    
    return images


def gamma_transform(image, gamma):
    #image = (image+1)/2.
    image = np.power(image, gamma)
    #image = image*2 - 1
    
    return image

def normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    image = (image-mean) / std
    
    return image

def gaussian_noise(image, mean, var):
    row, col, ch = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    
    noisy = image + gauss
    
    return noisy