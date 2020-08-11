"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import numpy as np
import tensorflow as tf
import keras

import keras.layers as KL


# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################

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


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])

    


############################################################
#  Resnet Graph
############################################################

def basic_conv(x, train_bn, filters, 
               kernel=3, stride=1, padding='same', 
               name='conv'):
    
    x = KL.Conv2D(filters, (kernel, kernel), 
                      strides=(stride, stride),
                      padding=padding,
                      name=name, use_bias=True)(x)
    x = BatchNorm(name=name+'_bn')(x, training=train_bn)
    x = KL.Activation(tf.nn.relu)(x)
    
    return x

   
def backbone(input_image, train_bn=None):     
    
    C1 = basic_conv(input_image,
                      train_bn, 64, 7, 2, 'valid', name='conv1')
    C2 = basic_conv(C1, 
                      train_bn, 128, 5, 2, 'same', name='conv2')
    C3 = basic_conv(C2, 
                      train_bn, 256, 3, 2, 'same', name='conv3') 
    C4 = basic_conv(C3, 
                      train_bn, 1024, 9, 1, 'same', name='conv4')
    C5 = basic_conv(C4, 
                      train_bn, 1024, 3, 1, 'same', name='conv5')
   
        
    up = KL.Lambda(lambda x: tf.nn.depth_to_space(x, 8, 'conv_up'), 
                   name='conv_up')(C5)
    up = KL.Activation(tf.nn.relu)(up)
        
    mask = KL.Conv2D(1, (3, 3), strides=1, 
                          padding = 'same', 
                          activation="sigmoid",
                          name = 'conv_mask')(up)
    
    return [C1, C2, C3, C4, C5, mask]