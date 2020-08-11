import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import keras.engine as KE


# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

from siammask import utils

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
    
    
class CustomUpscale2D(KL.Layer):   
    "Custom layer for depthwise correlation between kernel and search images."
    def __init__(self, size):        
        super(self.__class__, self).__init__()
        self.size = size
        
    def call(self, inputs):        
        out = tf.image.resize(inputs, self.size)
        
        return out
    
        
############################################################
#  Mask Head 
############################################################


def U_model(h_feature_map, u_feature_map, depth, features_in, batch_size, 
            name, up_kernel = 3, padding = 'valid'):
       
    h = KL.Conv2D(depth, (3, 3),
                  strides=(1, 1), padding = "same", activation = "relu",
                  name='mask_h' + name + '_conv1')(h_feature_map)
    h = KL.Conv2D(depth, (3, 3),
                  strides=(1, 1), padding = "same",
                  name='mask_h' + name + '_conv2')(h)    
    
    u = KL.Conv2D(depth*4, (3, 3),
                       strides=(1, 1), padding = "same", activation = "relu",
                       name='mask_u' + name + '_conv1')(u_feature_map)
    u = KL.Conv2D(depth*2, (3, 3),
                       strides=(1, 1), padding = "same", activation = "relu",
                       name='mask_u' + name + '_conv2')(u)
    u = KL.Conv2D(depth, (3, 3),
                       strides=(1, 1), padding = "same",
                       name='mask_u' + name + '_conv3')(u)   
    
    # reshape from [batch*height*width, h, w, channels] to [batch, height*width, h, w, channels]
    h = KL.Lambda(
        lambda t: tf.reshape(t, [batch_size, -1, 
                                 tf.shape(t)[1], tf.shape(t)[2], depth]))(h) 
    # reshape from [batch, h, w, channels] to [batch, 1, h, w, channels]
    u = KL.Lambda(
        lambda t: tf.expand_dims(t, 1))(u)
    
    add = KL.Add(name = 'mask_u' + name + '_add')([h, u])                             
    add = KL.Activation('relu')(add)
    
    # reshape from [batch, height*width, h, w, channels] to [batch*height*width, h, w, channels]
    add = KL.Lambda(
        lambda t: tf.reshape(t, [-1, tf.shape(t)[2], tf.shape(t)[3], depth]))(add) 
    
    up = KL.Conv2DTranspose(depth//2, up_kernel, strides = (2, 2),
                            padding = padding, 
                            output_padding = (0, 0),  
                            name = 'mask_u' + name + '_up')(add)
    
    return up
    


def mask_graph(corr_feature_map, kernel_feature_maps, config, level):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """       
    
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    x = KL.Conv2D(256, (1, 1),
                       strides=1,
                       name='mask_conv')(corr_feature_map)
    x = BatchNorm(name='mask_conv_bn')(x, training = config.TRAIN_BN)
    x = KL.Activation('relu')(x)   
   
    
    deconv_features = KL.Conv2DTranspose(32, 15, strides = (15, 15),
                                         name = "mask_deconv")(x)     

    u2 = U_model(deconv_features, C3, 
                 depth = 32, features_in = 512,
                 batch_size = config.BATCH_SIZE, name = "2", 
                 up_kernel = 3)
    u3 = U_model(u2, C2, 
                 depth = 16, features_in = 256, 
                 batch_size = config.BATCH_SIZE, name = "3", 
                 up_kernel = 3, padding = "same")
    u4 = U_model(u3, C1, 
                 depth = 8, features_in = 64,
                 batch_size = config.BATCH_SIZE, name = "4", 
                 up_kernel = 7)  
    
    rpn_mask = KL.Conv2D(1, (3, 3), strides=1, padding = 'same',
                            activation="sigmoid", name="rpn_mask")(u4)
    
    rpn_mask = KL.Lambda(
        lambda t: tf.reshape(t, [config.BATCH_SIZE, -1, 
                                 tf.shape(t)[1], tf.shape(t)[2], 1]))(rpn_mask) 

    return rpn_mask
    


def build_mask_model(config):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """    
    input_feature_maps = KL.Input(shape=[None, None, config.TOP_DOWN_PYRAMID_SIZE],
                                 name="input_mask_feature_maps")    
    C1 = KL.Input(shape=[None, None, 64],
                                 name="input_mask_refine_c1")
    C2 = KL.Input(shape=[None, None, 256],
                                 name="input_mask_refine_c2")
    C3 = KL.Input(shape=[None, None, 512],
                                 name="input_mask_refine_c3")      
    
   
    # deconv feature maps
    x = KL.Conv2D(256, (1, 1), strides=1, name='mask_conv')(input_feature_maps)
    x = BatchNorm(name='mask_conv_bn')(x, training = config.TRAIN_BN)
    x = KL.Activation('relu')(x)   
    deconv_features = KL.Conv2DTranspose(64, 16, strides = (16, 16), name = "mask_deconv")(x) 
    
     
    # U2
    u2 = KL.Conv2D(128, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u2_conv1')(C3)
    u2 = KL.Conv2D(64, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u2_conv2')(u2)
    u2 = KL.Conv2D(32, (3, 3), padding = "same",
                                   name='mask_u2_conv3')(u2)
    h2 = KL.Conv2D(32, (3, 3), padding = "same", activation = "relu",
                       name='mask_h2_conv1')(deconv_features)
    h2 = KL.Conv2D(32, (3, 3), padding = "same",
                       name='mask_h2_conv2')(h2)  
    u2 = KL.Add(name = 'mask_u2_add')([u2, h2])                             
    u2 = KL.Activation('relu')(u2)
    u2 = KL.Conv2DTranspose(16, 2, strides = (2, 2),
                            padding = "same",                           
                            name = 'mask_u2_up')(u2)
    
    # U3
    u3 = KL.Conv2D(64, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u3_conv1')(C2)
    u3 = KL.Conv2D(32, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u3_conv2')(u3)
    u3 = KL.Conv2D(16, (3, 3), padding = "same",
                                   name='mask_u3_conv3')(u3)
    h3 = KL.Conv2D(16, (3, 3), padding = "same", activation = "relu",
                       name='mask_h3_conv1')(u2)
    h3 = KL.Conv2D(16, (3, 3), padding = "same",
                       name='mask_h3_conv2')(h3)  
    u3 = KL.Add(name = 'mask_u3_add')([u3, h3])                             
    u3 = KL.Activation('relu')(u3)
    u3 = KL.Conv2DTranspose(8, 2, strides = (2, 2),
                            padding = "same",                           
                            name = 'mask_u3_up')(u3)
    
    # U4
    u4 = KL.Conv2D(32, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u4_conv1')(C1)
    u4 = KL.Conv2D(16, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u4_conv2')(u4)
    u4 = KL.Conv2D(8, (3, 3), padding = "same",
                                   name='mask_u4_conv3')(u4)
    h4 = KL.Conv2D(8, (3, 3), padding = "same", activation = "relu",
                       name='mask_h4_conv1')(u3)
    h4 = KL.Conv2D(8, (3, 3), padding = "same",
                       name='mask_h4_conv2')(h4)  
    u4 = KL.Add(name = 'mask_u4_add')([u4, h4])                             
    u4 = KL.Activation('relu')(u4)
    u4 = KL.Conv2DTranspose(4, 2, strides = (2, 2),
                            padding = "same",                           
                            name = 'mask_u4_up')(u4)
    
    rpn_masks = KL.Conv2D(1, (3, 3), strides=1, padding = 'same',
                            activation="sigmoid", name="rpn_masks")(u4) 
    
    return KM.Model([input_feature_maps, C1, C2, C3], rpn_masks, name="mask_model")


   
class MaskFeaturesLayer(KE.Layer):
    def __init__(self, shapes, config=None, **kwargs):
        super(MaskFeaturesLayer, self).__init__(**kwargs)
        self.config = config
        self.shapes = shapes
      

    def call(self, inputs):
        input_rpn_match = inputs[0]
        input_kernel_features = inputs[1:4]
        input_siammask_features = inputs[4:7]
        input_gt_masks = inputs[7]        
        siammask_features, target_masks = [], []
        k_f_1, k_f_2, k_f_3 = [], [], []
        
        
        if not self.config.ANCHORLESS:            
            input_rpn_match = tf.squeeze(input_rpn_match, -1)
            input_rpn_match = tf.reshape(input_rpn_match, [self.config.BATCH_SIZE, 
                                                           -1,
                                                           len(self.config.RPN_ANCHOR_RATIOS)])
            
        def get_features(rpn_match, kernel_features, siammask_features, target_mask):
            sf_shape = tf.shape(siammask_features)
            siammask_features = tf.reshape(siammask_features, [sf_shape[0], 
                                                               -1,
                                                               sf_shape[3]])        
            rpn_match = K.max(rpn_match, axis = 2)
            indices = tf.where(K.equal(rpn_match, 1))
            s_f = tf.gather_nd(siammask_features, indices)
            s_f = tf.reshape(s_f, [-1, 1, 1, sf_shape[3]]) 
            t_m = tf.gather_nd(target_mask, indices)
            
            kf1 =  tf.gather_nd(kernel_features[0], indices[:, :1])
            kf2 =  tf.gather_nd(kernel_features[1], indices[:, :1])
            kf3 =  tf.gather_nd(kernel_features[2], indices[:, :1])
            
            return s_f, t_m, kf1, kf2, kf3

            
        start = 0    
        for i in range(self.shapes.shape[0]):
            stop = start + (self.shapes[i][0]//2+1)*(self.shapes[i][1]//2+1)
            rpn_match = input_rpn_match[:, start:stop, :]
            start = stop
            
            # get target masks
            target_mask = tf.image.extract_image_patches(input_gt_masks, 
                                                         sizes = [1, self.config.KERNEL_IMAGE_SHAPE[0], self.config.KERNEL_IMAGE_SHAPE[1], 1],
                                                         strides = [1, self.config.BACKBONE_STRIDES[i], self.config.BACKBONE_STRIDES[i], 1],
                                                         rates = [1, 1, 1, 1],
                                                         padding = "VALID") 
            target_mask = tf.reshape(target_mask, [self.config.BATCH_SIZE, -1, 
                                                   self.config.KERNEL_IMAGE_SHAPE[0],
                                                   self.config.KERNEL_IMAGE_SHAPE[1],
                                                   1])            
            # get features
            s_f, t_m, kf1, kf2, kf3  = get_features(rpn_match, input_kernel_features,
                                                         input_siammask_features[i], target_mask)
            
            siammask_features.append(s_f)   
            target_masks.append(t_m)    
            k_f_1.append(kf1)
            k_f_2.append(kf2)
            k_f_3.append(kf3)

        
        siammask_features = tf.concat(siammask_features, axis = 0)
        target_masks = tf.concat(target_masks, axis = 0)
        k_f_1 = tf.concat(k_f_1, axis = 0)
        k_f_2 = tf.concat(k_f_2, axis = 0)
        k_f_3 = tf.concat(k_f_3, axis = 0)
        
        return [k_f_1, k_f_2, k_f_3, siammask_features, target_masks]
    
    
    def compute_output_shape(self, input_shape):
        return [
            (None, 64, 64, 64),  
            (None, 32, 32, 256),
            (None, 16, 16, 512),
            (None, 1, 1, self.config.TOP_DOWN_PYRAMID_SIZE),           
            (None, self.config.KERNEL_IMAGE_SHAPE[0], self.config.KERNEL_IMAGE_SHAPE[1], 1),           
            ] 
        
        
  
def mask_proposals_graph(rpn_match, features, input_gt_masks, config, nr_features):
    input_gt_masks = tf.expand_dims(input_gt_masks, 0) 
    gt_masks = []
    for i in range(nr_features):
        # get target masks      
        m = tf.image.extract_image_patches(input_gt_masks, 
                                          sizes = [1, config.KERNEL_IMAGE_SHAPE[0], config.KERNEL_IMAGE_SHAPE[1], 1],
                                          strides = [1, config.BACKBONE_STRIDES[i], config.BACKBONE_STRIDES[i], 1],
                                          rates = [1, 1, 1, 1],
                                          padding = "VALID")       
        m = tf.reshape(m, [-1, 
                           config.KERNEL_IMAGE_SHAPE[0],
                           config.KERNEL_IMAGE_SHAPE[1],
                           1]) 
        gt_masks.append(m)
        
    gt_masks = tf.concat(gt_masks, axis = 0)
        
    # get TRAIN_MASKS_PER_IMAGE random  masks
    rpn_match = tf.squeeze(rpn_match, -1)
    
    def get_indices(rpn_match, mask_per_img, val=1):
        indices = tf.where(K.equal(rpn_match, val))
        indices = tf.random.shuffle(indices)
        indices = indices[:mask_per_img, :]
        
        return indices
    
    indices_pos = get_indices(rpn_match, config.TRAIN_MASKS_PER_IMAGE, 1)
    indices_neg = get_indices(rpn_match, config.TRAIN_MASKS_PER_IMAGE-tf.shape(indices_pos)[0], 0)
    indices = tf.concat([indices_pos, indices_neg], axis = 0)
    
    features = tf.gather_nd(features, indices)
    features = tf.reshape(features, [-1, 1, 1, tf.shape(features)[-1]]) 
    gt_masks = tf.gather_nd(gt_masks, indices)
    rpn_match = tf.gather_nd(rpn_match, indices)    
    
    return features, gt_masks, rpn_match    

  
    
class MaskAnchorlessFeaturesLayer(KE.Layer):
    def __init__(self, config=None, shapes=None, **kwargs):
        super(MaskAnchorlessFeaturesLayer, self).__init__(**kwargs)
        self.config = config    
      

    def call(self, inputs):
        input_rpn_match = inputs[0]
        input_gt_masks = inputs[1]  
        input_siammask_features = inputs[2:]
                   
        
        for i in range(len(input_siammask_features)):
            f_shape = tf.shape(input_siammask_features[i])
            input_siammask_features[i] = tf.reshape(input_siammask_features[i],
                                                    [f_shape[0],
                                                     -1,
                                                     f_shape[3]])          
            
        siammask_features = tf.concat(input_siammask_features, axis = 1)
        
        siammask_features, target_masks, rpn_match = utils.batch_slice(
            [input_rpn_match, siammask_features, input_gt_masks],
            lambda x, y, z: mask_proposals_graph(x, y, z, self.config, len(input_siammask_features)),
            self.config.IMAGES_PER_GPU)    
        
        
        return [siammask_features, target_masks, rpn_match]
    
    
    def compute_output_shape(self, input_shape):
        return [           
            (self.config.BATCH_SIZE, self.config.TRAIN_MASKS_PER_IMAGE, 1, 1, self.config.TOP_DOWN_PYRAMID_SIZE),           
            (self.config.BATCH_SIZE, self.config.TRAIN_MASKS_PER_IMAGE, self.config.KERNEL_IMAGE_SHAPE[0], self.config.KERNEL_IMAGE_SHAPE[1], 1), 
            (self.config.BATCH_SIZE, self.config.TRAIN_MASKS_PER_IMAGE),
            ] 
        
        
    
class GetAnchorlessTargetMasks(KE.Layer):
    def __init__(self, config=None, **kwargs):
        super(GetAnchorlessTargetMasks, self).__init__(**kwargs)
        self.config = config    
      

    def call(self, inputs):
        input_gt_masks = inputs   
        k = 1                  
        if self.config.CROP:
            k = 2
        target_masks = tf.image.extract_image_patches(input_gt_masks, 
                                          sizes = [1, self.config.KERNEL_IMAGE_SHAPE[0]//k, self.config.KERNEL_IMAGE_SHAPE[1]//k, 1],
                                          strides = [1, self.config.BACKBONE_STRIDES[0], self.config.BACKBONE_STRIDES[0], 1],
                                          rates = [1, 1, 1, 1],
                                          padding = "VALID")       
        
        target_masks = tf.reshape(target_masks, [
                               self.config.BATCH_SIZE,
                               -1, 
                               self.config.KERNEL_IMAGE_SHAPE[0]//k,
                               self.config.KERNEL_IMAGE_SHAPE[1]//k,
                               1]) 
       
        
        return target_masks
    
    
    def compute_output_shape(self, input_shape):
        k = 1                  
        if self.config.CROP:
            k = 2
            
        return (self.config.BATCH_SIZE, 
                None, 
                self.config.KERNEL_IMAGE_SHAPE[0]//k, 
                self.config.KERNEL_IMAGE_SHAPE[1]//k, 
                1)           
            
                

    

def build_mask_anchorless_model(config):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """    
    input_feature_maps = KL.Input(
            shape=[None, 1, 1, config.TOP_DOWN_PYRAMID_SIZE],
            name="input_mask_feature_maps")    
    C1 = KL.Input(shape=[None, None, 64],
                                 name="input_mask_refine_c1")
    C2 = KL.Input(shape=[None, None, 256],
                                 name="input_mask_refine_c2")
    C3 = KL.Input(shape=[None, None, 512],
                                 name="input_mask_refine_c3")

     
    # deconv feature maps
    x = KL.TimeDistributed(KL.Conv2D(256, (1, 1), strides=1), 
                           name='mask_conv')(input_feature_maps)
    x = KL.TimeDistributed(BatchNorm(), 
                           name='mask_conv_bn')(x, training = config.TRAIN_BN)
    x = KL.Activation('relu')(x)   
    deconv_features = KL.TimeDistributed(                            
            KL.Conv2DTranspose(32, 16, strides = (16, 16)),
            name = "mask_deconv")(x)    
     
    # U2
    u2 = KL.Conv2D(128, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u2_conv1')(C3)
    u2 = KL.Conv2D(32, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u2_conv2')(u2)
    
    u2 = KL.Lambda(
        lambda t: tf.expand_dims(t, 1))(u2)
    h2 = KL.TimeDistributed(
            KL.Conv2D(32, (3, 3), padding = "same", activation = "relu"),
            name='mask_h2_conv1')(deconv_features)
    h2 = KL.TimeDistributed(
            KL.Conv2D(32, (3, 3), padding = "same", activation = "relu"),
            name='mask_h2_conv2')(h2)  
    u2 = KL.Add(name = 'mask_u2_add')([u2, h2])                             
    u2 = KL.TimeDistributed(
            KL.UpSampling2D(size=(2, 2)),                                      
            name = 'mask_u2_up')(u2)
    
    u2 = KL.TimeDistributed(
            KL.Conv2D(16, 3, strides = (1, 1), padding="same"),                           
            name = 'mask_u2_post')(u2)
   
    
    # U3
    u3 = KL.Conv2D(64, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u3_conv1')(C2)
    u3 = KL.Conv2D(16, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u3_conv2')(u3)

    u3 = KL.Lambda(
        lambda t: tf.expand_dims(t, 1))(u3)
    h3 = KL.TimeDistributed(
            KL.Conv2D(16, (3, 3), padding = "same", activation = "relu"),
            name='mask_h3_conv1')(u2)
    h3 = KL.TimeDistributed(
            KL.Conv2D(16, (3, 3), padding = "same", activation = "relu"),
            name='mask_h3_conv2')(h3)  
    u3 = KL.Add(name = 'mask_u3_add')([u3, h3])                             
    u3 = KL.TimeDistributed(
            KL.UpSampling2D(size=(2, 2)),                           
            name = 'mask_u3_up')(u3)
            
    u3 = KL.TimeDistributed(
            KL.Conv2D(4, 3, strides = (1, 1), padding="same"),                           
            name = 'mask_u3_post')(u3)
    
    # U4
    u4 = KL.Conv2D(16, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u4_conv1')(C1)
    u4 = KL.Conv2D(4, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u4_conv2')(u4)    
    u4 = KL.Lambda(
        lambda t: tf.expand_dims(t, 1))(u4)
    h4 = KL.TimeDistributed(
            KL.Conv2D(4, (3, 3), padding = "same", activation = "relu"),
            name='mask_h4_conv1')(u3)
    h4 = KL.TimeDistributed(
            KL.Conv2D(4, (3, 3), padding = "same", activation = "relu"),
            name='mask_h4_conv2')(h4)  
    u4 = KL.Add(name = 'mask_u4_add')([u4, h4])                             
    u4 = KL.TimeDistributed(
            KL.UpSampling2D(size=(2, 2)),                          
            name = 'mask_u4_up')(u4)
    
    rpn_masks_logits = KL.TimeDistributed(
                    KL.Conv2D(1, (3, 3), strides=1, 
                              padding = 'same'), 
                    name="rpn_masks_logits")(u4)
    rpn_masks = KL.Activation(
            "sigmoid", name="rpn_masks")(rpn_masks_logits)
    
    return KM.Model([input_feature_maps, C1, C2, C3], [rpn_masks_logits, rpn_masks], name="mask_model")


def build_mask_anchorless_unet_model(config):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """    
    input_feature_maps = KL.Input(shape=[None, 1, 1, config.TOP_DOWN_PYRAMID_SIZE],
                                 name="input_mask_feature_maps")    
    C1 = KL.Input(shape=[None, None, 64],
                                 name="input_mask_refine_c1")
    C2 = KL.Input(shape=[None, None, 256],
                                 name="input_mask_refine_c2")
       
    # deconv feature maps
    x = KL.TimeDistributed(KL.Conv2D(256, (1, 1), strides=1), 
                           name='mask_conv')(input_feature_maps)
    x = KL.TimeDistributed(BatchNorm(), 
                           name='mask_conv_bn')(x, training = config.TRAIN_BN)
    x = KL.Activation('relu')(x)   
    deconv_features = KL.TimeDistributed(                            
            KL.Conv2DTranspose(64, 32, strides = (32, 32)),
            name = "mask_deconv")(x)        
       
    # U3
    u3 = KL.Conv2D(64, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u3_conv1')(C2)
    u3 = KL.Conv2D(32, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u3_conv2')(u3)
    u3 = KL.Conv2D(16, (3, 3), padding = "same",
                                   name='mask_u3_conv3')(u3)
    u3 = KL.Lambda(
        lambda t: tf.expand_dims(t, 1))(u3)
    h3 = KL.TimeDistributed(
            KL.Conv2D(16, (3, 3), padding = "same", activation = "relu"),
            name='mask_h3_conv1')(deconv_features)
    h3 = KL.TimeDistributed(
            KL.Conv2D(16, (3, 3), padding = "same"),
            name='mask_h3_conv2')(h3)  
    u3 = KL.Add(name = 'mask_u3_add')([u3, h3])                             
    u3 = KL.Activation('relu')(u3)
    u3 = KL.TimeDistributed(
            #CustomUpscale2D([61, 61]),            
            KL.Conv2DTranspose(8, 3, strides = (2, 2), 
                               padding="same"),                           
            name = 'mask_u3_up')(u3)
    
    # U4
    u4 = KL.Conv2D(32, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u4_conv1')(C1)
    u4 = KL.Conv2D(16, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u4_conv2')(u4)
    u4 = KL.Conv2D(8, (3, 3), padding = "same",
                                   name='mask_u4_conv3')(u4)
    u4 = KL.Lambda(
        lambda t: tf.expand_dims(t, 1))(u4)
    h4 = KL.TimeDistributed(
            KL.Conv2D(8, (3, 3), padding = "same", activation = "relu"),
            name='mask_h4_conv1')(u3)
    h4 = KL.TimeDistributed(
            KL.Conv2D(8, (3, 3), padding = "same"),
            name='mask_h4_conv2')(h4)  
    u4 = KL.Add(name = 'mask_u4_add')([u4, h4])                             
    u4 = KL.Activation('relu')(u4)
    u4 = KL.TimeDistributed(
            #CustomUpscale2D([127, 127]),
            KL.Conv2DTranspose(4, 7, strides = (2, 2), 
                               padding="same"),
                               #output_padding = (1, 1)),                           
            name = 'mask_u4_up')(u4)
    
    rpn_masks = KL.TimeDistributed(
                    KL.Conv2D(1, (3, 3), strides=1, 
                              padding = 'same', 
                              activation="sigmoid"), 
                    name="rpn_masks")(u4) 
    
    return KM.Model([input_feature_maps, C1, C2], rpn_masks, name="mask_unet_model")




def build_mask_upsample_model(config):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """    
    input_feature_maps = KL.Input(shape=[None, 1, 1, config.TOP_DOWN_PYRAMID_SIZE],
                                 name="input_mask_feature_maps")    
    C1 = KL.Input(shape=[None, None, 64],
                                 name="input_mask_refine_c1")
    C2 = KL.Input(shape=[None, None, 256],
                                 name="input_mask_refine_c2")
       
    # deconv feature maps
    x = KL.TimeDistributed(KL.Conv2D(256, (1, 1), strides=1), 
                           name='mask_conv')(input_feature_maps)
    x = KL.TimeDistributed(BatchNorm(), 
                           name='mask_conv_bn')(x, training = config.TRAIN_BN)
    x = KL.Activation('relu')(x)   
    deconv_features = KL.TimeDistributed(                            
            KL.Conv2DTranspose(64, 32, strides = (32, 32)),
            name = "mask_deconv")(x)        
       
    # U3
    u3 = KL.Conv2D(64, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u3_conv1')(C2)
    u3 = KL.Conv2D(32, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u3_conv2')(u3)
    u3 = KL.Conv2D(16, (3, 3), padding = "same",
                                   name='mask_u3_conv3')(u3)
    u3 = KL.Lambda(
        lambda t: tf.expand_dims(t, 1))(u3)
    h3 = KL.TimeDistributed(
            KL.Conv2D(16, (3, 3), padding = "same", activation = "relu"),
            name='mask_h3_conv1')(deconv_features)
    h3 = KL.TimeDistributed(
            KL.Conv2D(16, (3, 3), padding = "same"),
            name='mask_h3_conv2')(h3)  
    u3 = KL.Add(name = 'mask_u3_add')([u3, h3])                             
    u3 = KL.Activation('relu')(u3)
    u3 = KL.TimeDistributed(
            #CustomUpscale2D([61, 61]),            
            KL.Conv2DTranspose(8, 3, strides = (2, 2), 
                               padding="same"),                           
            name = 'mask_u3_up')(u3)
    
    # U4
    u4 = KL.Conv2D(32, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u4_conv1')(C1)
    u4 = KL.Conv2D(16, (3, 3), padding = "same", activation = "relu",
                                   name='mask_u4_conv2')(u4)
    u4 = KL.Conv2D(8, (3, 3), padding = "same",
                                   name='mask_u4_conv3')(u4)
    u4 = KL.Lambda(
        lambda t: tf.expand_dims(t, 1))(u4)
    h4 = KL.TimeDistributed(
            KL.Conv2D(8, (3, 3), padding = "same", activation = "relu"),
            name='mask_h4_conv1')(u3)
    h4 = KL.TimeDistributed(
            KL.Conv2D(8, (3, 3), padding = "same"),
            name='mask_h4_conv2')(h4)  
    u4 = KL.Add(name = 'mask_u4_add')([u4, h4])                             
    u4 = KL.Activation('relu')(u4)
    u4 = KL.TimeDistributed(
            #CustomUpscale2D([127, 127]),
            KL.Conv2DTranspose(4, 7, strides = (2, 2), 
                               padding="same"),
                               #output_padding = (1, 1)),                           
            name = 'mask_u4_up')(u4)
    
    rpn_masks = KL.TimeDistributed(
                    KL.Conv2D(1, (3, 3), strides=1, 
                              padding = 'same', 
                              activation="sigmoid"), 
                    name="rpn_masks")(u4) 
    
    return KM.Model([input_feature_maps, C1, C2], rpn_masks, name="mask_unet_model")


                