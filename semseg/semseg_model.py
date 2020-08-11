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
import skimage
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from tensorflow.python.keras.layers import Lambda
from keras_lr_multiplier import LRMultiplier

import matplotlib.pyplot as plt


from semseg import utils, resnet, detection, losses, data_generator as dg
from semseg import config

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
        input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")        
     
        if mode == "training":
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)             
            input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype="uint8")
           
        # Get mask prediction       
        C1, C2, C3, C4, C5, rpn_masks = resnet.backbone(input_image,
                                    train_bn=config.TRAIN_BN)   
       
       
        if mode == "training":           
            rpn_mask_loss = KL.Lambda(lambda x: losses.rpn_mask_loss_graph(*x), name="rpn_mask_loss")(
                            [input_gt_masks, rpn_masks])           
            
            inputs_2d = [input_image, input_gt_boxes, input_gt_masks]              
            outputs_2d = [rpn_masks, rpn_mask_loss]                        
           
            model = KM.Model(inputs_2d, outputs_2d, name='sem_seg')
        
        
        else:      
            inputs_2d = [input_image]              
            outputs_2d = [rpn_masks] 
            
            model = KM.Model(inputs_2d, outputs_2d, name='sem_seg')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)   
       
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
        
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_mask_loss"]            
    
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
        train_generator = data_generator(train_imgs, train_masks, self.config, shuffle=True,                                         
                                         batch_size=self.config.BATCH_SIZE, mode = self.mode)
        val_generator = data_generator(val_imgs, val_masks, self.config, shuffle=True,
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
    
        
        
    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)       
        
        molded_images = np.stack(images, axis = 0)
        # Run object detection
        rpn_mask = self.keras_model.predict([molded_images], verbose=0)
        # Process detections
        results = []
        
        def get_centroid(mask):
            mask_thresh = np.round(mask).astype('uint8')
            properties = skimage.measure.regionprops(mask_thresh, mask)
            
            if properties:
                w_center = properties[0].weighted_centroid
            else:
                w_center = (mask.shape[0]//2, mask.shape[1]//2)
            
            return w_center
        
        for i, image in enumerate(images):
            rois = utils.extract_bboxes(np.round(rpn_mask[i]))
            centroid = get_centroid(rpn_mask[i, ..., 0])
            results.append({
                "centroid": centroid,    
                "rois": rois,
                "masks": rpn_mask[i],
            })
        return results 
    
    
    def test(self, images, labs = None, verbose = 1):
        assert self.mode == "inference", "Create model in inference mode."
        assert 1 == self.config.BATCH_SIZE, "Set BATCH_SIZE to 1."
        gt_bboxs, gt_masks = [], []
        pred_centroid, pred_bboxs, pred_masks = [], [], []
        gt, pred = {}, {}
        
        for i in range(images.shape[0]):  
            if labs is not None:
                image, gt_bbox, gt_mask = dg.load_image_gt(config,
                                                           images[i], labs[i],                                               
                                                           mode = self.mode)
                gt_bboxs.append(gt_bbox)
                gt_masks.append(gt_mask)
            else:
                image, _, _ = dg.load_image_gt(config,
                                               images[i], None,                                               
                                               mode = self.mode)
            
            result = self.detect([image], verbose)  
            pred_centroid.append(result[0]['centroid'])
            pred_bboxs.append(result[0]['rois'])
            pred_masks.append(result[0]['masks'])           
            
            
            print(i)
            print(result[0]['centroid'])
                       
                
        if labs is not None:    
            gt['gt_bboxs'] = np.stack(gt_bboxs, axis = 0)
            gt['gt_masks'] = np.stack(gt_masks, axis = 0)
        
        pred['pred_centroid'] = np.stack(pred_centroid, axis = 0)
        pred['pred_bboxs'] = np.stack(pred_bboxs, axis = 0)
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
        

  




def data_generator(imgs, masks, config, shuffle=True, batch_size=5, mode="training"):
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
    
    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
           
            # If the image source is not to be augmented pass None as augmentation
            input_image,\
            gt_boxes,\
            gt_masks  = dg.load_image_gt( config,
                                          imgs[image_id],
                                          masks[image_id],                                            
                                          mode=mode
                                          )       

            # Init batch arrays
            if b == 0:               
                batch_images = np.zeros(
                    (batch_size,) + input_image.shape, dtype=np.float32)  
                batch_gt_boxes = np.zeros(
                    (batch_size, 1, 4), dtype=np.int32)                  
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     1), dtype=np.float32)                     
                         

            # Add to batch                
            batch_images[b] = input_image.astype(np.float32)
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
           
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_gt_boxes, batch_gt_masks]
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
