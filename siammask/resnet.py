"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import numpy as np
import random
import tensorflow  as tf
import keras

import keras.backend as K
import keras.layers as KL
import keras.models as KM


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
def conv2d_bn_shared(x1, x2, _shared_layers, use_bn, train_bn, filters, 
               kernel=3, stride=1, padding='same', 
               conv_name='conv', bn_name='bn', dilation=1):
    
    if conv_name not in _shared_layers.keys():
        _shared_layers[conv_name] = KL.Conv2D(filters, (kernel, kernel), 
                      strides=(stride, stride),
                      padding=padding,
                      dilation_rate = dilation,
                      name=conv_name, use_bias=True)
    x1 = _shared_layers[conv_name](x1)
    x2 = _shared_layers[conv_name](x2)
    
    if use_bn:
        if bn_name not in _shared_layers.keys():
            _shared_layers[bn_name] = BatchNorm(name=bn_name, momentum=0.7)
            
        if random.randint(0,1) == 1:
            x1 = _shared_layers[bn_name](x1, training=train_bn)
            x2 = _shared_layers[bn_name](x2, training=train_bn)
        else:
            x2 = _shared_layers[bn_name](x2, training=train_bn)
            x1 = _shared_layers[bn_name](x1, training=train_bn)
       
    return x1, x2


def conv2d_bn(x, _shared_layers, use_bn, train_bn, filters, 
               kernel=3, stride=1, padding='same', 
               conv_name='conv', bn_name='bn', dilation=1):
    
    if conv_name not in _shared_layers.keys():
        _shared_layers[conv_name] = KL.Conv2D(filters, (kernel, kernel), 
                      strides=(stride, stride),
                      padding=padding,
                      dilation_rate = dilation,
                      name=conv_name, use_bias=True)
    x = _shared_layers[conv_name](x)
    
    if use_bn:
        if bn_name not in _shared_layers.keys():
            _shared_layers[bn_name] = BatchNorm(name=bn_name, momentum=0.7)
        x = _shared_layers[bn_name](x, training=train_bn)
   
       
    return x


def conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, filters, 
               kernel=3, stride=1, padding='same', 
               conv_name='conv', bn_name='bn', dilation=1):
    
    if conv_name not in _shared_layers.keys():
        _shared_layers[conv_name] = KL.Conv2D(filters, (kernel, kernel), 
                      strides=(stride, stride),
                      padding=padding,
                      dilation_rate = dilation,
                      name=conv_name, use_bias=True)
    x = _shared_layers[conv_name](x)
    
    if use_bn:
        if bn_name not in _shared_layers.keys():
            _shared_layers[bn_name] = BatchNorm(name=bn_name, momentum=0.7)
        x = _shared_layers[bn_name](x, training=train_bn)
 
    
    x = KL.Activation(tf.nn.relu)(x)
    

    return x



def conv2d_relu(x, _shared_layers, use_dp, filters, 
               kernel=3, stride=1, padding='same', 
               conv_name='conv', dilation=1):
    
    if conv_name not in _shared_layers.keys():
        _shared_layers[conv_name] = KL.Conv2D(filters, (kernel, kernel), 
                      strides=(stride, stride),
                      padding=padding,
                      dilation_rate = dilation,
                      name=conv_name, use_bias=True)
    x = _shared_layers[conv_name](x)
   
    x = KL.Activation(tf.nn.relu)(x)
    
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
    
    return x


def bn_relu_conv2d(x, _shared_layers, use_bn, train_bn, filters, 
               kernel=3, stride=1, padding='same', 
               conv_name='conv', bn_name ='bn', dilation=1):
    
    if use_bn:
        if bn_name not in _shared_layers.keys():
            _shared_layers[bn_name] = BatchNorm(name=bn_name, momentum=0.7)
        x = _shared_layers[bn_name](x, training=train_bn)    
    
    x = KL.Activation(tf.nn.relu)(x)
    
    if conv_name not in _shared_layers.keys():
        _shared_layers[conv_name] = KL.Conv2D(filters, (kernel, kernel), 
                      strides=(stride, stride),
                      padding=padding,
                      dilation_rate = dilation,
                      name=conv_name, use_bias=True)
    x = _shared_layers[conv_name](x)    
    
    
    return x


# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, _shared_layers, kernel_size, filters, stage, block,
                   dilation=1, use_bias=True, use_bn=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'  
    
    # 1 x 1
    x = conv2d_bn_relu(input_tensor, _shared_layers, use_bn, train_bn, nb_filter1, 
               kernel=1, stride=1, padding='valid', 
               conv_name=conv_name_base + '2a', 
               bn_name=bn_name_base + '2a',
               dilation=1)
    # 3 x 3
    x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, nb_filter2, 
               kernel=kernel_size, stride=1, padding='same', 
               conv_name=conv_name_base + '2b', 
               bn_name=bn_name_base + '2b',
               dilation=dilation)    
        
    # 1 x 1
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, nb_filter3, 
               kernel=1, stride=1, padding='valid', 
               conv_name=conv_name_base + '2c', 
               bn_name=bn_name_base + '2c',
               dilation=1)    
    
    x = KL.Add()([x, input_tensor])    
    x = KL.Activation(tf.nn.relu)(x)
    
    return x



def conv_block(input_tensor, _shared_layers, kernel_size, filters, stage, block,
               strides=(2, 2), dilation = 1, use_bias=True, use_bn=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """ 
    padding='same'
    if strides[0]==2:
        padding='same'
        
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'   
          
   
    # 1 x 1
    x = conv2d_bn_relu(input_tensor, _shared_layers, use_bn, train_bn, nb_filter1, 
               kernel=1, stride=1, padding='valid', 
               conv_name=conv_name_base + '2a', 
               bn_name=bn_name_base + '2a',
               dilation=1)    
    
    # 3 x 3
    x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, nb_filter2, 
               kernel=kernel_size, stride=strides[0], padding=padding, 
               conv_name=conv_name_base + '2b', 
               bn_name=bn_name_base + '2b',
               dilation=dilation) 
        
    # 1 x 1
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, nb_filter3, 
               kernel=1, stride=1, padding='valid', 
               conv_name=conv_name_base + '2c', 
               bn_name=bn_name_base + '2c',
               dilation=1)         
      
    # 1 x 1 shortcut
    shortcut = conv2d_bn(input_tensor, _shared_layers, use_bn, train_bn, nb_filter3, 
               kernel=1, stride=strides[0], padding=padding, 
               conv_name=conv_name_base + '1', 
               bn_name=bn_name_base + '1',
               dilation=1)    
    
    
    x = KL.Add()([x, shortcut])   
    x = KL.Activation(tf.nn.relu)(x)
    
    return x



def basic_identity_block(input_tensor, _shared_layers, kernel_size, filters, stage, block,
               strides=(1, 1), dilation = 1, use_bias=True, use_bn=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """ 
            
    nb_filter1, nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'   
          
   
    # 3 x 3
    x = conv2d_bn_relu(input_tensor, _shared_layers, use_bn, train_bn, nb_filter1, 
               kernel=kernel_size, stride=strides[0], padding='same', 
               conv_name=conv_name_base + '2a', 
               bn_name=bn_name_base + '2a',
               dilation=dilation)    
    
    # 3 x 3
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, nb_filter2, 
               kernel=kernel_size, stride=1, padding='same', 
               conv_name=conv_name_base + '2b', 
               bn_name=bn_name_base + '2b',
               dilation=1)    
    
    
    x = KL.Add()([x, input_tensor])   
    x = KL.Activation(tf.nn.relu)(x)
    
    return x



def basic_conv_block(input_tensor, _shared_layers, kernel_size, filters, stage, block,
               strides=(1, 1), dilation = 1, use_bias=True, use_bn=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """ 
            
    nb_filter1, nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'   
          
   
    # 3 x 3
    x = conv2d_bn_relu(input_tensor, _shared_layers, use_bn, train_bn, nb_filter1, 
               kernel=kernel_size, stride=strides[0], padding='same', 
               conv_name=conv_name_base + '2a', 
               bn_name=bn_name_base + '2a',
               dilation=dilation)    
    
    # 3 x 3
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, nb_filter2, 
               kernel=kernel_size, stride=1, padding='same', 
               conv_name=conv_name_base + '2b', 
               bn_name=bn_name_base + '2b',
               dilation=1) 
        
    
    # 1 x 1 shortcut
    shortcut = conv2d_bn(input_tensor, _shared_layers, use_bn, train_bn, nb_filter2, 
               kernel=1, stride=strides[0], padding='valid', 
               conv_name=conv_name_base + '1', 
               bn_name=bn_name_base + '1',
               dilation=1)    
    
    
    x = KL.Add()([x, shortcut])   
    x = KL.Activation(tf.nn.relu)(x)
    
    return x


def ci_identity_block(input_tensor, _shared_layers, kernel_size, filters, stage, block,
               strides=(1, 1), dilation = 1, use_bias=True, use_bn=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """ 
            
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'   
          
   
    # 1 x 1
    x = conv2d_bn_relu(input_tensor, _shared_layers, use_bn, train_bn, nb_filter1, 
               kernel=1, stride=1, padding='valid', 
               conv_name=conv_name_base + '2a', 
               bn_name=bn_name_base + '2a',
               dilation=1)    
    
    # 3 x 3
    x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, nb_filter2, 
               kernel=kernel_size, stride=strides[0], padding="same", 
               conv_name=conv_name_base + '2b', 
               bn_name=bn_name_base + '2b',
               dilation=dilation) 
        
    # 1 x 1
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, nb_filter3, 
               kernel=1, stride=1, padding='valid', 
               conv_name=conv_name_base + '2c', 
               bn_name=bn_name_base + '2c',
               dilation=1)         
      
        
    x = KL.Add()([x, input_tensor])   
    x = KL.Activation(tf.nn.relu)(x)
    
    x = KL.Cropping2D(cropping=((1,1), (1,1)))(x)
    
    return x


def ci_conv_block(input_tensor, _shared_layers, kernel_size, filters, stage, block,
               strides=(1, 1), dilation = 1, use_bias=True, use_bn=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """ 
            
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'   
          
   
    # 1 x 1
    x = conv2d_bn_relu(input_tensor, _shared_layers, use_bn, train_bn, nb_filter1, 
               kernel=1, stride=1, padding='valid', 
               conv_name=conv_name_base + '2a', 
               bn_name=bn_name_base + '2a',
               dilation=1)    
    
    # 3 x 3
    x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, nb_filter2, 
               kernel=kernel_size, stride=1, padding="same", 
               conv_name=conv_name_base + '2b', 
               bn_name=bn_name_base + '2b',
               dilation=dilation) 
        
    # 1 x 1
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, nb_filter3, 
               kernel=1, stride=1, padding='valid', 
               conv_name=conv_name_base + '2c', 
               bn_name=bn_name_base + '2c',
               dilation=1)         
      
    # 1 x 1 shortcut
    shortcut = conv2d_bn(input_tensor, _shared_layers, use_bn, train_bn, nb_filter3, 
               kernel=1, stride=1, padding="valid", 
               conv_name=conv_name_base + '1', 
               bn_name=bn_name_base + '1',
               dilation=1)    
    
    
    x = KL.Add()([x, shortcut])   
    x = KL.Activation(tf.nn.relu)(x)
    
    x = KL.Cropping2D(cropping=((1,1), (1,1)))(x)
    if strides[0] == 2:
        x = KL.MaxPooling2D((2, 2), strides=(2, 2), padding="valid")(x) 
    
    return x



    
def ciresnet22(input_image, _shared_layers, use_bn=True, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """    
    # Stage 1
    # Stage 1
    if 'conv1' not in _shared_layers.keys():
        _shared_layers['conv1'] = KL.Conv2D(64, (7, 7), strides=(2, 2),
                  padding='valid', name='conv1', use_bias=True)
    x = _shared_layers['conv1'](input_image)
    
    if use_bn:
        if 'bn_conv1' not in _shared_layers.keys():
            _shared_layers['bn_conv1'] = BatchNorm(name='bn_conv1')
        x = _shared_layers['bn_conv1'](x, training=train_bn)   
    x = KL.Activation(tf.nn.relu)(x)
    
    C1 = x    
    # Stage 2
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = ci_conv_block(x, _shared_layers, 3, [64, 64, 256], stage=2, block='a', use_bn=use_bn, train_bn=train_bn)
    x = ci_identity_block(x, _shared_layers, 3, [64, 64, 256], stage=2, block='b', use_bn=use_bn, train_bn=train_bn)
    C2 = x = ci_identity_block(x, _shared_layers, 3, [64, 64, 256], stage=2, block='c', use_bn=use_bn, train_bn=train_bn)
    
    # Stage 3
    x = ci_conv_block(x, _shared_layers, 3, [128, 128, 512], stage=3, block='a', strides=(2, 2), use_bn=use_bn, train_bn=train_bn)
    x = ci_identity_block(x, _shared_layers, 3, [128, 128, 512], stage=3, block='b', use_bn=use_bn, train_bn=train_bn)
    x = ci_identity_block(x, _shared_layers, 3, [128, 128, 512], stage=3, block='c', dilation=2, use_bn=use_bn, train_bn=train_bn)
    C3 = x = ci_identity_block(x, _shared_layers, 3, [128, 128, 512], stage=3, block='d', use_bn=use_bn, train_bn=train_bn)
    
    C4 = C3
       
    return [C1, C2, C3, C4]


def resnet50(input_image, _shared_layers, stage4=False, use_bn=True, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """    
    
    # Stage 1
    if 'conv1' not in _shared_layers.keys():
        _shared_layers['conv1'] = KL.Conv2D(64, (7, 7), strides=(2, 2),
                  padding='valid', name='conv1', use_bias=True)
    x = _shared_layers['conv1'](input_image)
    
    if use_bn:
        if 'bn_conv1' not in _shared_layers.keys():
            _shared_layers['bn_conv1'] = BatchNorm(name='bn_conv1')
        x = _shared_layers['bn_conv1'](x, training=train_bn)   
    x = KL.Activation(tf.nn.relu)(x)
    
    C1 = x    
    # Stage 2
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = conv_block(x, _shared_layers, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), use_bn=use_bn, train_bn=train_bn)
    x = identity_block(x, _shared_layers, 3, [64, 64, 256], stage=2, block='b', use_bn=use_bn, train_bn=train_bn)
    C2 = x = identity_block(x, _shared_layers, 3, [64, 64, 256], stage=2, block='c', use_bn=use_bn, train_bn=train_bn)
    
    # Stage 3
    x = conv_block(x, _shared_layers, 3, [128, 128, 512], stage=3, block='a', use_bn=use_bn, train_bn=train_bn)
    x = identity_block(x, _shared_layers, 3, [128, 128, 512], stage=3, block='b', use_bn=use_bn, train_bn=train_bn)
    x = identity_block(x, _shared_layers, 3, [128, 128, 512], stage=3, block='c', use_bn=use_bn, train_bn=train_bn)
    C3 = x = identity_block(x, _shared_layers, 3, [128, 128, 512], stage=3, block='d', use_bn=use_bn, train_bn=train_bn)
    
    # Stage 4
    if stage4:
        x = conv_block(x, _shared_layers, 3, [256, 256, 1024], stage=4, block='a', strides = (1, 1), dilation = 2, use_bn=use_bn, train_bn=train_bn)
        block_count = 5
        for i in range(block_count):
            x = identity_block(x, _shared_layers, 3, [256, 256, 1024], stage=4, block=chr(98 + i),
                               dilation=1, use_bn=use_bn, train_bn=train_bn)          
        C4 = x   
    else:
        C4 = None  
       
    return [C1, C2, C3, C4]



def resnet18(input_image, _shared_layers, stage4=True, use_bn=True, train_bn=True):
    # Stage 1
    if 'conv1' not in _shared_layers.keys():
        _shared_layers['conv1'] = KL.Conv2D(64, (7, 7), strides=(2, 2),
                  padding='valid', name='conv1', use_bias=True)
    x = _shared_layers['conv1'](input_image)
    
    if use_bn:
        if 'bn_conv1' not in _shared_layers.keys():
            _shared_layers['bn_conv1'] = BatchNorm(name='bn_conv1')
        x = _shared_layers['bn_conv1'](x, training=train_bn)  
    x = KL.Activation(tf.nn.relu)(x)
   
    C1 = x    
    # Stage 2
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = basic_conv_block(x, _shared_layers, 3, [64, 64], stage=2, block='a', use_bn=use_bn, train_bn=train_bn)
    C2 = x = basic_identity_block(x, _shared_layers, 3, [64, 64], stage=2, block='b', use_bn=use_bn, train_bn=train_bn)
    
    # Stage 3
    x = basic_conv_block(x, _shared_layers, 3, [128, 128], stage=3, block='a', strides=(2, 2), use_bn=use_bn, train_bn=train_bn)
    C3 = x = basic_identity_block(x, _shared_layers, 3, [128, 128], stage=3, block='b', use_bn=use_bn, train_bn=train_bn)
    
    # Stage 4
    if stage4:
        x = basic_conv_block(x, _shared_layers, 3, [256, 256], stage=4, block='a', dilation = 2, use_bn=use_bn, train_bn=train_bn)
        C4 = x = basic_identity_block(x, _shared_layers, 3, [256, 256], stage=4, block='b', use_bn=use_bn, train_bn=train_bn)

    else:
        C4 = None  
       
    return [C1, C2, C3, C4]
    


def alexnet(input_image, _shared_layers, stage4=True, use_bn=True, train_bn=True, use_dp=False):
    # Stage 1
    x = conv2d_bn(input_image, _shared_layers, use_bn, train_bn, 96, 
                   kernel=11, stride=2, padding='valid', 
                   conv_name='conv1', bn_name='conv1_bn')
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x)
    x = KL.Activation(tf.nn.relu)(x)
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
    C1 = x
    # Satge 2
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, 256, 
                   kernel=5, stride=1, padding='valid', 
                   conv_name='conv2', bn_name='conv2_bn')
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x)
    x = KL.Activation(tf.nn.relu)(x)
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
    C2 = x
    # Stage 3
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, 384, 
                   kernel=3, stride=1, padding='valid', 
                   conv_name='conv3', bn_name='conv3_bn')
    x = KL.Activation(tf.nn.relu)(x)
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, 384, 
                   kernel=3, stride=1, padding='valid', 
                   conv_name='conv4', bn_name='conv4_bn')
    x = KL.Activation(tf.nn.relu)(x)
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
    C3 = x
    # Stage 
    if stage4:
        x = conv2d_bn(x, _shared_layers, use_bn, train_bn, 256, 
                       kernel=3, stride=1, padding='valid', 
                       conv_name='conv5', bn_name='conv5_bn')
        x = KL.Activation(tf.nn.relu)(x)
        if use_dp:
            x = KL.Dropout(rate=0.1)(x)
        C4 = x
    else:
        C4 = None
    
    return [C1, C2, C3, C4]

def alexnet_deeper(input_image, _shared_layers, stage4=True, use_bn=True, train_bn=True, use_dp=False):
    # Stage 1
    x = conv2d_bn(input_image, _shared_layers, use_bn, train_bn, 96, 
                   kernel=11, stride=2, padding='valid', 
                   conv_name='conv1', bn_name='conv1_bn')
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x)
    x = KL.Activation(tf.nn.relu)(x)
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
    C1 = x
    # Satge 2
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, 256, 
                   kernel=5, stride=1, padding='valid', 
                   conv_name='conv2', bn_name='conv2_bn')
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x)
    x = KL.Activation(tf.nn.relu)(x)
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
    C2 = x
    # Stage 3
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, 512, 
                   kernel=3, stride=1, padding='valid', 
                   conv_name='conv3', bn_name='conv3_bn')
    x = KL.Activation(tf.nn.relu)(x)
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, 512, 
                   kernel=3, stride=1, padding='valid', 
                   conv_name='conv4', bn_name='conv4_bn')
    x = KL.Activation(tf.nn.relu)(x)
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
    C3 = x
    # Stage 
    if stage4:
        x = conv2d_bn(x, _shared_layers, use_bn, train_bn, 1024, 
                       kernel=3, stride=1, padding='valid', 
                       conv_name='conv5', bn_name='conv5_bn')
        x = KL.Activation(tf.nn.relu)(x)
        if use_dp:
            x = KL.Dropout(rate=0.1)(x)
        C4 = x
    else:
        C4 = None
    
    return [C1, C2, C3, C4]


def alexnet_padding(input_image, _shared_layers, stage4=True, use_bn=True, train_bn=True, use_dp=False):
    # Stage 1
    x = conv2d_bn(input_image, _shared_layers, use_bn, train_bn, 96, 
                   kernel=11, stride=2, padding='same', 
                   conv_name='conv1', bn_name='conv1_bn')
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = KL.Activation(tf.nn.relu)(x)
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
    C1 = x
    # Satge 2
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, 256, 
                   kernel=5, stride=1, padding='same', 
                   conv_name='conv2', bn_name='conv2_bn')
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = KL.Activation(tf.nn.relu)(x)
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
    C2 = x
    # Stage 3
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, 384, 
                   kernel=3, stride=1, padding='same', 
                   conv_name='conv3', bn_name='conv3_bn')
    x = KL.Activation(tf.nn.relu)(x)    
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
        
    x = conv2d_bn(x, _shared_layers, use_bn, train_bn, 384, 
                   kernel=3, stride=1, padding='same', 
                   conv_name='conv4', bn_name='conv4_bn')
    x = KL.Activation(tf.nn.relu)(x)
    if use_dp:
        x = KL.Dropout(rate=0.1)(x)
    C3 = x
    
    # Stage 
    if stage4:
        x = conv2d_bn(x, _shared_layers, use_bn, train_bn, 256, 
                       kernel=3, stride=1, padding='same', 
                       conv_name='conv5', bn_name='conv5_bn')
        x = KL.Activation(tf.nn.relu)(x)
        if use_dp:
            x = KL.Dropout(rate=0.1)(x)
        C4 = x
    else:
        C4 = None
    
    return [C1, C2, C3, C4]



def vgg(input_image, _shared_layers, stage4=True, use_bn=True, train_bn=True):
    # Stage 1
    x = conv2d_bn_relu(input_image, _shared_layers, use_bn, train_bn, 64, 
                       kernel=3, stride=1, padding='valid', 
                       conv_name='conv1', bn_name='conv1_bn')
    x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, 64, 
                       kernel=3, stride=1, padding='valid', 
                       conv_name='conv2', bn_name='conv2_bn')
    C1 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), padding="valid")(x)
    # Satge 2
    x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, 128, 
                       kernel=3, stride=1, padding='valid', 
                       conv_name='conv3', bn_name='conv3_bn')
    x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, 128, 
                       kernel=3, stride=1, padding='valid', 
                       conv_name='conv4', bn_name='conv4_bn')
    C2 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), padding="valid")(x)
    # Stage 3
    x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn,  256, 
                       kernel=3, stride=1, padding='valid', 
                       conv_name='conv5', bn_name='conv5_bn')
    x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, 256, 
                       kernel=3, stride=1, padding='valid', 
                       conv_name='conv6', bn_name='conv6_bn')    
    x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, 256, 
                       kernel=3, stride=1, padding='valid', 
                       conv_name='conv7', bn_name='conv7_bn')
    C3 = x = KL.MaxPooling2D((2, 2), strides=(2, 2), padding="valid")(x)
    
    # Stage 
    if stage4:
        x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, 512, 
                       kernel=3, stride=1, padding='valid', 
                       conv_name='conv8', bn_name='conv8_bn')
        x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, 512, 
                           kernel=3, stride=1, padding='valid', 
                           conv_name='conv9', bn_name='conv9_bn')
        C4 = x = conv2d_bn_relu(x, _shared_layers, use_bn, train_bn, 512, 
                               kernel=3, stride=1, padding='valid', 
                               conv_name='conv10', bn_name='conv10_bn')
    else:
        C4 = None
    
    return [C1, C2, C3, C4]




def backbone(input_image, _shared_layers, config, train_bn):
    if config.BACKBONE == 'resnet50':
        return resnet50(input_image, _shared_layers,
                            stage4=config.STAGE4,
                            use_bn=config.USE_BN,
                            train_bn=train_bn)
        
    elif config.BACKBONE == 'resnet18':
        return resnet18(input_image, _shared_layers,
                            stage4=config.STAGE4,
                            use_bn=config.USE_BN,
                            train_bn=train_bn)
    
    elif config.BACKBONE == 'alexnet':
        return alexnet(input_image, _shared_layers,
                            stage4=config.STAGE4,
                            use_bn=config.USE_BN,
                            train_bn=train_bn,
                            use_dp = config.USE_DP)
    
    elif config.BACKBONE == 'alexnet_deeper':
        return alexnet_deeper(input_image, _shared_layers,
                            stage4=config.STAGE4,
                            use_bn=config.USE_BN,
                            train_bn=train_bn,
                            use_dp = config.USE_DP)  
        
    elif config.BACKBONE == 'alexnet_padding':
        return alexnet_padding(input_image, _shared_layers,
                            stage4=config.STAGE4,
                            use_bn=config.USE_BN,
                            train_bn=train_bn,
                            use_dp = config.USE_DP)    
           
        
    elif config.BACKBONE == 'vgg':
        return vgg(input_image, _shared_layers,
                            stage4=config.STAGE4,
                            use_bn=config.USE_BN,
                            train_bn=train_bn)
   
    elif config.BACKBONE == 'ciresnet22':
        return ciresnet22(input_image, _shared_layers,
                          use_bn=config.USE_BN,
                          train_bn=train_bn)
    else:
        raise NotImplementedError
    
  
def backbone_graph(_shared_layers, config):
    input_img = KL.Input(shape=(None, None, config.IMAGE_CHANNEL_COUNT))
    
    outputs = backbone(input_img, _shared_layers, config)
    
    return KM.Model([input_img], outputs, name='backbone')
    
    