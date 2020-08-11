"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow  as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from tensorflow.python.keras.layers import Lambda
from keras_lr_multiplier import LRMultiplier

import matplotlib.pyplot as plt


from siammask import utils, resnet, rpn as rp, mask, proposal, detection, losses, data_generator as dg
from siammask import config

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

"""
import dicom_to_numpy as dtn
import numpy as np



# LOAD SOME DATA
load_data_path = 'D:/MasterAIThesis/h5py_data/patient1/'
imgs = dtn.load_data(load_data_path + 'X_data_normalized.hdf5', 'X_data_normalized')[0]
masks = dtn.load_data(load_data_path + 'ROI_data.hdf5', 'ROI_data')[0]
image, mask = imgs[310], masks[310]

config = config.Config()

image, image_meta, class_ids, bbox, mask = dg.load_image_gt(image, mask, 2, [1], config)

"""


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
    
    
class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

class DepthwiseCorr(KL.Layer):   
    "Custom layer for depthwise correlation between kernel and search images."
    def __init__(self, name):        
        super(self.__class__, self).__init__()
        self.name = name
        
    def call(self, inputs):        
        kernel = inputs[0]
        search = inputs[1]        
       
        search_shape = tf.shape(search)
        kernel_shape = tf.shape(kernel)
        batch_size, s_h, s_w, s_c = search_shape[0], search_shape[1], search_shape[2], search_shape[3]
        k_h, k_w, k_c = kernel_shape[1], kernel_shape[2], kernel_shape[3]
        search = tf.transpose(search, [1, 2, 0, 3])
        kernel = tf.transpose(kernel, [1, 2, 0, 3])
        search = tf.reshape(search,
                            [1, s_h, s_w, batch_size * s_c])
        kernel = tf.reshape(kernel,
                            [k_h, k_w, batch_size * k_c, 1])
        out = tf.nn.depthwise_conv2d(search, kernel, [1, 1, 1, 1], 'VALID', name = self.name)
        out = tf.reshape(out, [tf.shape(out)[1], tf.shape(out)[2], batch_size, k_c])
        out = tf.transpose(out, [2, 0, 1, 3])
        
        return out


    
class CustomUpscale2D(KL.Layer):   
    "Custom layer for depthwise correlation between kernel and search images."
    def __init__(self, size):        
        super(self.__class__, self).__init__()
        self.size = size
        
    def call(self, inputs):        
        out = tf.image.resize(inputs, self.size)
        
        return out
    


############################################################
#  MaskRCNN Class
############################################################

class SiamMask():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.shared_layers = {}
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Inputs
        input_kernel_image = KL.Input(
            shape=[None, None, config.KERNEL_IMAGE_SHAPE[2]], name="input_kernel_image")
        input_search_image = KL.Input(
            shape=[None, None, config.SEARCH_IMAGE_SHAPE[2]], name="input_search_image")        
        input_search_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_search_image_meta")
        input_bbox_centers = KL.Input(
                    shape=[None, 2], name="input_bbox_centers", dtype=tf.float32)
       
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)             
            input_rpn_centerness = KL.Input(
                    shape=[None, 1], name="input_rpn_centerness", dtype=tf.float32)
           
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_search_image)[1:3]))(input_gt_boxes)
             
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            input_kernel_gt_masks = KL.Input(
                    shape=[config.KERNEL_IMAGE_SHAPE[0], config.KERNEL_IMAGE_SHAPE[1], None],
                    name="input_kernel_gt_masks", dtype="uint8")
            input_search_gt_masks = KL.Input(
                    shape=[config.SEARCH_IMAGE_SHAPE[0], config.SEARCH_IMAGE_SHAPE[1], None],
                    name="input_search_gt_masks", dtype="uint8")
        
        
        # Build the shared convolutional layers.
        # Bottom-up Layers
        
        C1_k, C2_k, C3_k, C4_k = resnet.backbone(input_kernel_image,
                                                 _shared_layers = self.shared_layers,
                                                 config=config,
                                                 train_bn=config.TRAIN_BN)            
        C1_s, C2_s, C3_s, C4_s = resnet.backbone(input_search_image,
                                                 _shared_layers = self.shared_layers,
                                                 config=config,
                                                 train_bn=config.TRAIN_BN)
   
            
        
        """
        backbone = resnet.backbone_graph(config)
        C1_k, C2_k, C3_k, C4_k = backbone([input_kernel_image])        
        C1_s, C2_s, C3_s, C4_s = backbone([input_search_image])
        """
        
        C_k = C3_k
        C_s = C3_s
        if config.STAGE4:
            C_k = C4_k
            C_s = C4_s
        
        # RPN Model
        # generate rpn predictions 
        P_k_class, P_s_class, P_k_bbox, P_s_bbox,\
        P_k_s_class, P_k_s_bbox,\
        rpn_class_logits, rpn_class, rpn_bbox,\
        rpn_centerness_logits, rpn_centerness,\
        rpn_masks_logits, rpn_masks = rp.adjust_depthwise_corr_layer(C_k, C_s, self.shared_layers, 4, config)
        
        if mode=='inference':
            rpn_features = [rpn_class_logits, rpn_class, rpn_bbox,\
                            rpn_centerness_logits, rpn_centerness]
            
            
            rpn_class_logits, rpn_class, rpn_bbox,\
            rpn_centerness_logits, rpn_centerness = proposal.LocalizationLayer(
                                                level=0,
                                                shape=config.FEATURES_SHAPE,
                                                config=config)(rpn_features)
                       
        rpn_mask_model = mask.build_mask_anchorless_model(config)        

        if mode == "training":            
            if config.MASK:
                if not config.REFINE:
                    target_masks = mask.GetAnchorlessTargetMasks(
                                config = config, 
                                name="target_masks")(input_search_gt_masks)
                                        
                    rpn_mask_loss = KL.Lambda(lambda x: losses.rpn_mask_loss_graph(config, *x), name="rpn_mask_loss")(
                        [target_masks, input_rpn_match, rpn_masks_logits])              
                    
                if config.REFINE:
                    mask_feature_maps,\
                    target_masks,\
                    rpn_match = mask.MaskAnchorlessFeaturesLayer(
                            config = config, 
                            name="mask_features")([
                                                input_rpn_match, 
                                                input_search_gt_masks,
                                                P_k_s_class])                                         
                    rpn_masks_logits, rpn_masks = rpn_mask_model([mask_feature_maps, C1_k, C2_k, C3_k])
                                                          
                    rpn_mask_loss = KL.Lambda(lambda x: losses.rpn_mask_refine_loss_graph(config, *x), name="rpn_mask_loss")(
                        [target_masks, rpn_match, rpn_masks_logits])              
                    
                
            # Losses               
            rpn_class_loss = KL.Lambda(lambda x: losses.rpn_class_anchorless_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            if config.ANCHORLESS:
                rpn_bbox_loss = KL.Lambda(lambda x: losses.rpn_bbox_anchorless_loss_graph(config, *x), name="rpn_bbox_loss")(
                    [input_rpn_bbox, input_rpn_match, input_bbox_centers, rpn_bbox])
            else:
                rpn_bbox_loss = KL.Lambda(lambda x: losses.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                    [input_rpn_bbox, input_rpn_match, rpn_bbox])
            rpn_centerness_loss = KL.Lambda(lambda x: losses.rpn_centerness_loss_graph(*x), name="rpn_centerness_loss")(
                [input_rpn_centerness, input_rpn_match, rpn_centerness_logits])                
           
            
        
            # Model
            inputs_2d = [input_kernel_image, input_search_image, 
                         input_search_image_meta, input_bbox_centers,                     
                         input_rpn_match, input_rpn_bbox,
                         input_rpn_centerness, 
                         input_gt_boxes, 
                         input_kernel_gt_masks,
                         input_search_gt_masks]  
            
            if config.MASK:
                outputs_2d = [P_k_class, P_s_class, P_k_bbox, P_s_bbox,
                              P_k_s_class,
                              C1_k, C2_k, C3_k, 
                              C1_s, C2_s, C3_s,                          
                              rpn_masks_logits, rpn_masks, target_masks, 
                              rpn_class_logits, rpn_class, rpn_bbox,
                              rpn_centerness_logits, rpn_centerness,  
                              rpn_class_loss, rpn_bbox_loss,                              
                              rpn_centerness_loss,
                              rpn_mask_loss
                              ]     
            else:
                outputs_2d = [P_k_class, P_s_class, P_k_bbox, P_s_bbox,
                              P_k_s_class,
                              C1_k, C2_k, C3_k,
                              C1_s, C2_s, C3_s,                         
                              rpn_class_logits, rpn_class, rpn_bbox,
                              rpn_centerness_logits, rpn_centerness,  
                              rpn_class_loss, rpn_bbox_loss,                              
                              rpn_centerness_loss
                              ]     
                    
       
            model = KM.Model(inputs_2d, outputs_2d, name='siam_mask')    
    
        else:           
            if config.MASK:
                if not config.REFINE:
                    pred_class,\
                    pred_bbox,\
                    pred_masks,\
                    max_detections,\
                    max_masks,\
                    penalty = detection.AnchorlessDetectionLayerMask(
                            backbone_shapes = config.FEATURES_SHAPE,
                            config=config, name = "detection_layer")(
                                    [rpn_class, rpn_centerness, 
                                     rpn_bbox, rpn_masks,
                                     input_bbox_centers, 
                                     input_search_image_meta
                                     ])                           
                    
                else:
                    pred_class,\
                    pred_bbox,\
                    max_detections,\
                    penalty,\
                    mask_feature_maps,\
                    max_bbox_centers = detection.AnchorlessDetectionLayerRefine(
                            backbone_shapes = config.FEATURES_SHAPE,
                            config=config, 
                            name = "detection_layer")(
                                    [rpn_class, rpn_centerness, 
                                     rpn_bbox,
                                     input_bbox_centers, 
                                     input_search_image_meta,
                                     P_k_s_class                                        
                                     ]) 
                    _, pred_masks = rpn_mask_model([mask_feature_maps, C1_k, C2_k, C3_k])

                    max_masks = KL.Lambda(lambda a: utils.batch_slice(
                            [a[0], a[1]],
                            lambda x, y: detection.postprocess_max_mask(x, y, config),
                            config.IMAGES_PER_GPU))([pred_masks, max_bbox_centers])                                  

                outputs = [pred_class, pred_bbox, pred_masks,                               
                              max_detections, max_masks]
                                      
               
            else:
                pred_class,\
                pred_bbox,\
                max_detections,\
                penalty = detection.AnchorlessDetectionLayer(
                        backbone_shapes = config.FEATURES_SHAPE,
                        config=config, name = "detection_layer")(
                                [rpn_class, rpn_centerness, 
                                 rpn_bbox,
                                 input_bbox_centers, 
                                 input_search_image_meta
                                 ])
                    
                outputs = [pred_class, pred_bbox,                               
                          max_detections]
          
            model = KM.Model([input_kernel_image, input_search_image,  
                              input_search_image_meta, input_bbox_centers],                        
                             outputs,
                             name='siam_mask')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)   
        """
        model = KM.Model([input_kernel_image, input_search_image, 
                             input_search_image_meta, input_bbox_centers,                     
                             input_rpn_match, input_rpn_bbox,
                             input_rpn_centerness, 
                             input_gt_boxes, input_gt_masks],                        
                         [C1_k, C2_k, C3_k, C4_k, 
                          C1_s, C2_s, C3_s, C4_s, 
                          ],
                          name='siam_mask')
        """
        return model
    
    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        #trainable_vars = model.keras_model.trainable_weights
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        
        lr_multiplier = {'conv': 0.5, 'res': 0.5, 'bn': 0.5}
        optimizer = LRMultiplier(optimizer, lr_multiplier)
        
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss", "rpn_bbox_loss", 
            "rpn_centerness_loss" ,
            "rpn_mask_loss"               
            ]   
        if not self.config.MASK:
            loss_names = [
            "rpn_class_loss", "rpn_bbox_loss", 
            "rpn_centerness_loss"
            ]   
        
        
        for name in loss_names:            
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))        
        

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)
            #self.keras_model.add_metric(loss, name=name, aggregation='mean')
        
        """
        if 'l2' not in self.keras_model.metrics_names:
            self.keras_model.metrics_names.append('l2')
            self.keras_model.metrics_tensors.append(tf.add_n(reg_losses))
        """   
            
    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers
            
        # Pre-defined layer regular expressions
        layer_regex = {
            # All layers
            "all": ".*",
            "mask": r"(conv\_.*)|(mask\_.*)",
            # all layers but the backbone
            "heads": r"(rpn\_.*)|(fpn\_.*)|(mask\_.*)",
            # From a specific Resnet stage and up
            "2+": r"(conv2.*)|(bn_conv2.*)|(conv3.*)|(bn_conv3.*)|(conv4.*)|(bn_conv4.*)|(conv5.*)|(bn_conv5.*)|(conv\_.*)|(block2.*)|(bn_block2.*)|(res2.*)|(bn2.*)|(block3.*)|(bn_block3.*)|(res3.*)|(bn3.*)|(block4.*)|(bn_block4.*)|(res4.*)|(bn4.*)|(block5.*)|(bn_block5.*)|(res5.*)|(bn5.*)|(dw\_.*)|(rpn\_.*)|(fpn\_.*)|(mask\_.*)",
            "3+": r"(conv3.*)|(bn_conv3.*)|(conv4.*)|(bn_conv4.*)|(conv5.*)|(bn_conv5.*)|(conv\_.*)|(block3.*)|(bn_block3.*)|(res3.*)|(bn3.*)|(block4.*)|(bn_block4.*)|(res4.*)|(bn4.*)|(block5.*)|(bn_block5.*)|(res5.*)|(bn5.*)|(dw\_.*)|(rpn\_.*)|(fpn\_.*)|(mask\_.*)",
            "4+": r"(conv4.*)|(bn_conv4.*)|(conv5.*)|(bn_conv5.*)|(conv\_.*)|(block4.*)|(bn_block4.*)|(res4.*)|(bn4.*)|(block5.*)|(bn_block5.*)|(res5.*)|(bn5.*)|(dw\_.*)|(rpn\_.*)|(fpn\_.*)|(mask\_.*)",
            "5+": r"(conv5.*)|(bn_conv5.*)|(conv\_.*)|(res5.*)|(bn5.*)|(dw\_.*)|(rpn\_.*)|(fpn\_.*)|(mask\_.*)",
        
        }                   
            
        # Exclude some layers  
        layers_show = layers
        if exclude:
            layers_show = filter(lambda l: not bool(re.fullmatch(layer_regex[exclude], l.name)), layers_show)
            layers = filter(lambda l: not bool(re.fullmatch(layer_regex[exclude], l.name)), layers)
                  
        indent = 0
        print("LAYERS LOADED: ")
        for layer in layers_show:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                indent=indent + 4
                continue

            if not layer.weights:
                continue            
            log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))           
        
        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()            
            
        
        # Update the log directory
        self.set_log_dir(filepath)
        
        
        
    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
        
        
        
    def train(self, train_imgs, train_masks, val_imgs, val_masks, learning_rate, epochs, layers, custom_callbacks=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # All layers
            "all": ".*",
            # all layers but the backbone
            "heads": r"(dw\_.*)|(rpn\_.*)|(fpn\_.*)|(mask\_.*)",
            # From a specific Resnet stage and up
            "2+": r"(conv2.*)|(bn_conv2.*)|(conv3.*)|(bn_conv3.*)|(conv4.*)|(bn_conv4.*)|(conv5.*)|(bn_conv5.*)|(conv\_.*)|(block2.*)|(bn_block2.*)|(res2.*)|(bn2.*)|(block3.*)|(bn_block3.*)|(res3.*)|(bn3.*)|(block4.*)|(bn_block4.*)|(res4.*)|(bn4.*)|(block5.*)|(bn_block5.*)|(res5.*)|(bn5.*)|(dw\_.*)|(rpn\_.*)|(fpn\_.*)|(mask\_.*)",
            "3+": r"(conv3.*)|(bn_conv3.*)|(conv4.*)|(bn_conv4.*)|(conv5.*)|(bn_conv5.*)|(conv\_.*)|(block3.*)|(bn_block3.*)|(res3.*)|(bn3.*)|(block4.*)|(bn_block4.*)|(res4.*)|(bn4.*)|(block5.*)|(bn_block5.*)|(res5.*)|(bn5.*)|(dw\_.*)|(rpn\_.*)|(fpn\_.*)|(mask\_.*)",
            "4+": r"(conv4.*)|(bn_conv4.*)|(conv5.*)|(bn_conv5.*)|(conv\_.*)|(block4.*)|(bn_block4.*)|(res4.*)|(bn4.*)|(block5.*)|(bn_block5.*)|(res5.*)|(bn5.*)|(dw\_.*)|(rpn\_.*)|(fpn\_.*)|(mask\_.*)",
            "5+": r"(conv5.*)|(bn_conv5.*)|(conv\_.*)|(block5.*)|(bn_block5.*)|(res5.*)|(bn5.*)|(dw\_.*)|(rpn\_.*)|(fpn\_.*)|(mask\_.*)",
        
        }    
        
        
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_imgs, train_masks, self.config, 2, np.array([1]), shuffle=True,                                         
                                         batch_size=self.config.BATCH_SIZE, mode = self.mode)
        val_generator = data_generator(val_imgs, val_masks, self.config, 2, np.array([1]), shuffle=True,
                                       batch_size=self.config.BATCH_SIZE, mode = self.mode)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=False,
        )
        self.epoch = max(self.epoch, epochs)              
    
        
        
    def detect(self, kernel_images, search_images, search_image_metas, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes        
        scores: [N] float probability scores for the class IDs
        masks: [N, H, W] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        #assert search_images.shape[0] == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(search_images.shape[0]))
            for i in range(search_images.shape[0]):
                log("image", search_images[i])
        
        backbone_shapes = compute_backbone_shapes(self.config, self.config.SEARCH_IMAGE_SHAPE)
        mappings = utils.generate_pyramid_mappings(backbone_shapes,
                                         self.config.BACKBONE_STRIDES,
                                         self.config.RPN_ANCHOR_STRIDE,
                                         self.config)
        mappings = np.broadcast_to(mappings, (search_images.shape[0],) + mappings.shape)

                
        results = []
        mask_shape = (search_images.shape[0],) + tuple(self.config.IMAGE_SHAPE[:2])
        max_masks = np.zeros(mask_shape[:3], dtype='uint8')
        if self.config.MASK:
            pred_class, pred_bboxs, pred_masks,\
            max_detections, max_masks =\
            self.keras_model.predict([kernel_images, search_images, 
                                      search_image_metas, mappings], verbose=0)
            
            pred_masks = pred_masks[..., 0]
            max_masks = max_masks[..., 0]
            max_masks = utils.unmold_anchorless_masks(self.config, 
                                                       max_detections, 
                                                       max_masks, 
                                                       search_image_metas)
           
        else:
            pred_class, pred_bboxs,\
            max_detections =\
            self.keras_model.predict([kernel_images, search_images, 
                                      search_image_metas, mappings], verbose=0)
            
        search_rois = np.round(max_detections[:, -4:]).astype('int32') 
                      
        max_detections = utils.unmold_anchorless_detections(max_detections, search_image_metas)
        results.append({
                "rois": max_detections[:, -4:].astype('int32'),  
                "search_rois": search_rois,
                "scores": max_detections[:, -5],
                "masks": max_masks
                
            })     
    
            
        return results 
    
    
    def test(self, images, labs=None, verbose = 1):
        assert self.mode == "inference", "Create model in inference mode."
        assert 1 == self.config.BATCH_SIZE, "Set BATCH_SIZE to 1."
        gt_bboxs, gt_search_bboxs, gt_masks = [], [], []
        pred_bboxs, pred_search_bboxs, pred_masks, pred_scores = [], [], [], [] 
        gt, pred = {}, {}
        
        kernel_image = images[0]
        if labs is not None:
            kernel_bbox = utils.extract_bboxes(labs[0, ..., np.newaxis])
        else:            
            shape = np.array(images.shape[1:])
            search_center = shape//2
            kernel_bbox = [search_center[0]-32, search_center[1]-32,
                           search_center[0]+32, search_center[1]+32]
            kernel_bbox = np.array(kernel_bbox, dtype='int32')[np.newaxis, ...]
           
                   
        for i in range(images.shape[0]):
            print(i)
            #kernel_bbox = utils.extract_bboxes(labs[i, ..., np.newaxis])
            
            if labs is not None:
                kernel_image_input, search_image, \
                search_image_meta,\
                _, search_mask, \
                search_bbox, original_bbox, _, _ = dg.load_image_gt(self.config, 
                                                    kernel_image, images[i], 
                                                    kernel_mask=None, search_mask=labs[i], 
                                                    kernel_bbox=kernel_bbox,
                                                    mode = self.mode)            
                        
            
                gt_bboxs.append(original_bbox)
                gt_search_bboxs.append(search_bbox)
                gt_masks.append(labs[i])           
                    
            else:
                kernel_image_input, search_image, \
                search_image_meta, _, \
                _, _, _, _, _ = dg.load_image_gt(self.config, 
                                        kernel_image, images[i], 
                                        kernel_mask=None, search_mask=None, 
                                        kernel_bbox=kernel_bbox,
                                        mode = self.mode)               
            
            if i > 0:
                result = self.detect(kernel_image_input[np.newaxis, ...], 
                                     search_image[np.newaxis, ...], 
                                     search_image_meta[np.newaxis, ...], verbose) 
                                                
                if(result[0]['scores'][0]>=0.3):
                    pred_bboxs.append(result[0]['rois'][0])
                else:
                    pred_bboxs.append(pred_bboxs[-1])
            
                pred_search_bboxs.append(result[0]['search_rois'][0])
                pred_scores.append(result[0]['scores'])
                pred_masks.append(result[0]['masks'][0])
                print(pred_bboxs[-1])               
                
                
            else:
                if labs is not None:
                    pred_bboxs.append(gt_bboxs[-1][0])
                    pred_search_bboxs.append(gt_search_bboxs[-1][0])
    
                    #pred_mask_bboxs.append(gt_bboxs[-1][0])
                    pred_scores.append(np.array([1], dtype = 'float32'))
                    pred_masks.append(gt_masks[-1])
                else:
                    pred_scores.append(np.array([1], dtype = 'float32'))
                    pred_bboxs.append(kernel_bbox[0])
                    pred_masks.append(np.zeros(images[0].shape[:2], dtype='uint8'))

            if pred_scores[-1][0] >= 0.3:    
                kernel_image = images[i]
            kernel_bbox = pred_bboxs[-1][np.newaxis, ...]
            #kernel_bbox = utils.extract_bboxes(labs[i, ..., np.newaxis])

            
        if labs is not None:
            gt['gt_bboxs'] = np.stack(gt_bboxs, axis = 0)
            gt['gt_search_bboxs'] = np.stack(gt_search_bboxs, axis = 0)
            gt['gt_masks'] = np.stack(gt_masks, axis = 0)
        
        pred['pred_bboxs'] = np.stack(pred_bboxs, axis = 0)
        pred['pred_search_bboxs'] = np.stack(pred_search_bboxs, axis = 0)

        #pred['pred_mask_bboxs'] = np.stack(pred_mask_bboxs, axis = 0)
        pred['pred_scores'] = np.stack(pred_scores, axis = 0)
        pred['pred_masks'] = np.stack(pred_masks, axis = 0)

        return gt, pred
         
            
    
    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        image_metas: If provided, the images are assumed to be already
            molded (i.e. resized, padded, and normalized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np           
        

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a     
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        
        return self._anchor_cache[tuple(image_shape)]




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


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)




def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """

    # Currently supports ResNet only
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])
                
                
def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    original_image_shape = meta[:, 0:3]
    image_shape = meta[:, 3:6]
    window = meta[:, 6:10]  # (y1, x1, y2, x2) window of image in in pixels    
    active_class_ids = meta[:, 10:]
    return {       
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),       
        "active_class_ids": active_class_ids.astype(np.int32),
    }


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

def data_generator(imgs, masks, config, num_classes, class_ids, shuffle=True, batch_size=5, mode="training"):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
  
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.
    no_augmentation_sources: Optional. List of sources to exclude for
        augmentation. A source is string that identifies a dataset and is
        defined in the Dataset class.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.arange(1, imgs.shape[1]-1)    
    error_count = 0
    
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.SEARCH_IMAGE_SHAPE)
    mappings = utils.generate_pyramid_mappings(backbone_shapes,
                                         config.BACKBONE_STRIDES,
                                         config.RPN_ANCHOR_STRIDE,
                                         config)       
    
    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            search_image_id_1 = image_ids[image_index]
            if random.randint(0, 1) == 1:
                kernel_image_id_1 = search_image_id_1 - random.randint(1, np.min([search_image_id_1, 5]))
            else:
                kernel_image_id_1 = search_image_id_1 + random.randint(1, np.min([len(image_ids)-search_image_id_1+1, 5]))
            search_image_id_0 = np.random.randint(imgs.shape[0])
            #kernel_image_id_0 = search_image_id_0
            #search_image_id_0 = 0
            #search_image_id_1 = 1
            #kernel_image_id_0 = 0
            #kernel_image_id_1 = 1
            kernel_image_id_0 = np.random.randint(imgs.shape[0])
            # If the image source is not to be augmented pass None as augmentation
            kernel_image, search_image, \
            search_image_meta, \
            kernel_gt_masks, search_gt_masks,\
            gt_boxes, _,\
            shift_x, shift_y  = dg.load_image_gt(config,
                                          imgs[kernel_image_id_0, kernel_image_id_1],
                                          imgs[search_image_id_0, search_image_id_1],
                                          kernel_mask = masks[kernel_image_id_0, kernel_image_id_1],                                                  
                                          search_mask = masks[search_image_id_0, search_image_id_1],
                                          mode=mode
                                          )
            
            # RPN Targets
            if config.ANCHORLESS:
                rpn_match, rpn_bbox,\
                rpn_centerness, _ = dg.build_anchorless_rpn_targets(mappings,
                                                       gt_boxes, config, 
                                                       shift_x, shift_y, config.BBOX_RATIO)
               
            else:                    
                rpn_match, _,\
                rpn_centerness, rpn_bbox = dg.build_anchorless_rpn_targets(mappings,
                                                       gt_boxes, config, 
                                                       shift_x, shift_y, config.BBOX_RATIO)
               
           
            # Init batch arrays
            if b == 0:                            
                batch_rpn_match = np.zeros(
                    [batch_size, mappings.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, mappings.shape[0], 4], dtype=np.float32)                    
                batch_rpn_centerness = np.zeros(
                    [batch_size, mappings.shape[0], 1], dtype=np.float32)
                batch_bbox_centers = np.zeros(
                    [batch_size, mappings.shape[0], 2], dtype=np.float32)
                
                
                batch_kernel_images = np.zeros(
                    (batch_size,) + kernel_image.shape, dtype=np.float32)
                batch_search_images = np.zeros(
                    (batch_size,) + search_image.shape, dtype=np.float32)
                batch_search_image_metas = np.zeros(
                    (batch_size,) + search_image_meta.shape, dtype=np.float32)
                batch_gt_boxes = np.zeros(
                    (batch_size, 1, 4), dtype=np.int32)
                batch_kernel_gt_masks = np.zeros(
                    (batch_size, kernel_gt_masks.shape[0], kernel_gt_masks.shape[1],
                     1), dtype=np.float32)                     
                batch_search_gt_masks = np.zeros(
                    (batch_size, search_gt_masks.shape[0], search_gt_masks.shape[1],
                     1), dtype=np.float32)           

            # Add to batch           
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_rpn_centerness[b] = rpn_centerness
            batch_bbox_centers[b] = mappings                
               
            batch_kernel_images[b] = kernel_image.astype(np.float32)
            batch_search_images[b] = search_image.astype(np.float32)  
            batch_search_image_metas[b] = search_image_meta.astype(np.float32) 
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_kernel_gt_masks[b, :, :, :kernel_gt_masks.shape[-1]] = kernel_gt_masks
            batch_search_gt_masks[b, :, :, :search_gt_masks.shape[-1]] = search_gt_masks
           
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_kernel_images, batch_search_images,  
                              batch_search_image_metas, batch_bbox_centers, 
                              batch_rpn_match, batch_rpn_bbox, batch_rpn_centerness,                                   
                              batch_gt_boxes, 
                              batch_kernel_gt_masks, batch_search_gt_masks]                    
            
                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image            
            error_count += 1
            if error_count > 5:
                raise
