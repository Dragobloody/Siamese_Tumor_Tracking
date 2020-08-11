"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import tensorflow as tf
import keras
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
    
    

class InstanceNorm(KL.Layer):
    def __init__(self, name):        
        super(self.__class__, self).__init__()
        
    def call(self, inputs):        
        out = tf.contrib.layers.instance_norm(inputs, trainable=True)
        
        return out
    
    
class DepthwiseCorr(KL.Layer):   
    "Custom layer for depthwise correlation between kernel and search images."
    def __init__(self, name):    
        super(self.__class__, self).__init__(name=name)
        #self.name = name
        self.output_dims = (256,)
     
    def build(self, input_shape):
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dims),
                                    initializer='zeros',
                                    trainable=True)
    
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

        out = tf.nn.bias_add(out, self.bias)
        
        return out    

############################################################
#  Region Proposal Network (RPN)
############################################################



#-------------------------------ANCHORLESS-------------------------------

def adjustment_layer(x, _shared_layers, shared=True, 
                     filters=256, kernel=(1, 1), 
                     padding="SAME", name='fpn', 
                     use_bn = True,
                     train_bn=False,                     
                     use_activation=True):
    
    if shared:
        if name not in _shared_layers.keys():
            _shared_layers[name] = KL.Conv2D(filters, 
                                             kernel,                       
                                             padding=padding,
                                             name = name)                                
                               
        x = _shared_layers[name](x)
        if use_bn:
            if name+'_bn' not in _shared_layers.keys():
                _shared_layers[name+'_bn'] = BatchNorm(name=name+'_bn', momentum=0.7)
            x = _shared_layers[name+'_bn'](x, training=train_bn)        
        
    else:
        x = KL.Conv2D(filters, kernel, padding=padding, 
                              name=name)(x)
        if use_bn:
            x = BatchNorm(name=name+'_bn', momentum=0.7)(x, training=train_bn)
        
    
    if use_activation:
        x = KL.Activation(tf.nn.relu)(x)
        
       
    return x

def adjust_corr_head(P_k, P_s, _shared_layers, k_size, config, stage, name='class'):
    if config.SHARED:
        name_k = 'fpn_p'+str(stage) + '_' + name + str(stage)
        name_s = name_k
    else:
        name_k = 'fpn_p'+str(stage) +'k_' + name + str(stage)
        name_s = 'fpn_p'+str(stage) +'s_' + name + str(stage)
    
    # adjust features for cross-correlation
    P_k_a = adjustment_layer(P_k, _shared_layers, config.SHARED,
                             filters=256, kernel=(k_size, k_size),
                             name=name_k, 
                             use_bn=config.USE_BN,
                             train_bn=config.TRAIN_BN,
                             use_activation=False)   
   
    P_s_a = adjustment_layer(P_s, _shared_layers, config.SHARED,
                             filters=256, kernel=(k_size, k_size),
                             name=name_s, 
                             use_bn=config.USE_BN,
                             train_bn=config.TRAIN_BN,
                             use_activation=False)
    
     
            
        
    # depthwise cross-correlation
    P_k_s = DepthwiseCorr(name = 'dw_p'+str(stage)+'ks_' + name)([P_k_a, P_s_a])
    
    return P_k_a, P_s_a, P_k_s

   
def adjust_depthwise_corr_layer(C_k, C_s, _shared_layers, stage, config):
    P_k = C_k
    P_s = C_s
    if config.SIAMRPN:    
        # bring channels nr to 256
        P_k = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), 
                        name='fpn_c'+ str(stage) + 'p'+str(stage)+'k')(C_k)
        P_s = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), 
                        name='fpn_c'+ str(stage) + 'p'+str(stage)+'s')(C_s)
    
    if config.CROP:
        # crop the central 8x8 region
        P_k = KL.Cropping2D(cropping=((4, 4), (4, 4)))(P_k)
    
    # depthwise cross-correlation
    P_k_class, P_s_class, P_k_s_class = adjust_corr_head(P_k, P_s, _shared_layers, 
                                                         3, config, stage, name='class')   
    P_k_bbox, P_s_bbox, P_k_s_bbox = adjust_corr_head(P_k, P_s, _shared_layers, 
                                                      3, config, stage, name='bbox')
    if config.MASK and not config.REFINE:
         P_k_mask, P_s_mask, P_k_s_mask = adjust_corr_head(P_k, P_s, _shared_layers,
                                      3, config, stage, name='mask')
    
#--------------------------- CLASSIFICATION HEAD ----------------------------------------
    # classification
    x = KL.Conv2D(256, (1, 1),
                       strides = (1, 1),
                       padding = 'same',
                       name='rpn_conv1_class' + str(stage))(P_k_s_class)
    
    x = BatchNorm(name='rpn_conv1_bn_class' + str(stage),
                  momentum=0.9)(x, training=None)   
    x = KL.Activation('relu')(x)  

    if config.USE_DP:
        x = KL.Dropout(rate=0.5)(x)      
    
    rpn_class_logits = KL.Conv2D(2, (1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 name='rpn_class_logits' + str(stage))(x)
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(rpn_class_logits)
    rpn_probs = KL.Activation(
        "sigmoid", name="rpn_class")(rpn_class_logits)
    
#-------------------------------- MASK HEAD ----------------------------------------
    # mask
    rpn_masks_logits, rpn_masks = None, None
    if config.MASK and not config.REFINE:
        x = KL.Conv2D(256, (1, 1),
                       strides = (1, 1),
                       padding = 'same',
                       name='rpn_conv1_mask' + str(stage))(P_k_s_mask)
        
        x = BatchNorm(name='rpn_conv1_bn_mask' + str(stage), 
                      momentum=0.9)(x, training=None)       
        x = KL.Activation('relu')(x)
        
        if config.USE_DP:
            x = KL.Dropout(rate=0.5)(x)
        
        rpn_masks_logits = KL.Conv2D(64*64, (1, 1),
                                     strides = (1, 1),
                                     padding = 'same',                                 
                                     name='rpn_masks_logits' + str(stage))(x)
        rpn_masks_logits = KL.Lambda(
            lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 64, 64, 1]))(rpn_masks_logits)
        
        if not config.CROP:
            rpn_masks_logits = KL.TimeDistributed(
                    KL.UpSampling2D(size=(2, 2)), 
                    name='rpn_masks_logits_up' + str(stage))(rpn_masks_logits)
        
        rpn_masks = KL.Activation(
            "sigmoid", name="rpn_masks")(rpn_masks_logits)
        
    
#--------------------------- REGRESSION + CENTERNESS HEAD ----------------------------------------
    # shared Conv2D
    x = KL.Conv2D(256, (1, 1),
                       strides = (1, 1),
                       padding = 'same',
                       name='rpn_conv1_bbox' + str(stage))(P_k_s_bbox)
    
    x = BatchNorm(name='rpn_conv1_bn_bbox' + str(stage), 
                  momentum=0.9)(x, training=None)    
    x = KL.Activation('relu')(x) 

    if config.USE_DP:
        x = KL.Dropout(rate=0.5)(x)      
    
    # regression
    rpn_bbox = KL.Conv2D(4, (1, 1),
                         strides = (1, 1),
                         padding = 'same',
                         activation='linear',
                         name='rpn_bbox' + str(stage))(x)  
    if config.ANCHORLESS:
        rpn_bbox = KL.Activation('relu')(rpn_bbox)
        #rpn_bbox = KL.Lambda(
        #    lambda t: tf.keras.activations.exponential(t))(rpn_bbox)  
    rpn_bbox = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(rpn_bbox)
    
    
    # centerness
    rpn_centerness_logits = KL.Conv2D(1, (1, 1),
                         strides = (1, 1),
                         padding = 'same',                       
                         name='rpn_centerness_logits'+ str(stage))(x)       
    # Reshape to [batch, h*w, 1]
    rpn_centerness_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 1]))(rpn_centerness_logits)
    
    rpn_centerness = KL.Activation(
        "sigmoid", name="rpn_centerness"+ str(stage))(rpn_centerness_logits)

   
    return [P_k_class, P_s_class, P_k_bbox, P_s_bbox,
            P_k_s_class, P_k_s_bbox,
            rpn_class_logits, rpn_probs,
            rpn_bbox,
            rpn_centerness_logits, rpn_centerness,
            rpn_masks_logits, rpn_masks]
