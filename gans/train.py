#! /usr/bin/python
# -*- coding: utf8 -*-
import sys
sys.path.append('D:/MasterAIThesis/code/load_dicom/')
sys.path.append('D:/MasterAIThesis/code/srgan/')
import os
import time
import random
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model import get_sr_G, get_dx_G, get_D, SRGAN_d2, get_patch_D, cycle_G
from config import config


import os
import dicom_to_numpy as dtn
import h5py
import numpy as np
import matplotlib.pyplot as plt

"""
###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
batch_size = 4
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128

# ni = int(np.sqrt(batch_size))

# create folders to save result images and trained models
save_dir = "samples"
tl.files.exists_or_mkdir(save_dir)
checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)
"""

def gamma_transform(image, gamma):
    image = (image+1)/2.
    image = np.power(image, gamma)
    image = image*2 - 1
    
    return image

"""
load_dir = 'D:/MasterAIThesis/code/srgan/dataset/'
drr_imgs = dtn.load_data(load_dir + 'drrs.hdf5', 'drrs')
xray_imgs = dtn.load_data(load_dir + 'xrays.hdf5', 'xrays')



drr_imgs = drr_imgs[:360, 128:, 128:-128]
xray_imgs = xray_imgs[:360, 128:, 128:-128]

c = 1/np.log(1 + np.max(drr_imgs, axis = (1,2)))
drr_imgs = np.log(drr_imgs+1)
drr_imgs = np.multiply(c[..., np.newaxis, np.newaxis], drr_imgs)
drr_imgs = (drr_imgs - np.min(drr_imgs))/(np.max(drr_imgs)-np.min(drr_imgs))
drr_imgs = drr_imgs*2 - 1

c = 1/np.log(1 + np.max(xray_imgs, axis = (1,2)))
xray_imgs = np.log(xray_imgs+1)
xray_imgs = np.multiply(c[..., np.newaxis, np.newaxis], xray_imgs)
xray_imgs = (xray_imgs - np.min(xray_imgs))/(np.max(xray_imgs)-np.min(xray_imgs))
xray_imgs = xray_imgs*2 - 1
xray_imgs = gamma_transform(xray_imgs,4)





#xray_imgs_gamma = gamma_transform(xray_imgs, 7)

drr_imgs = drr_imgs[..., np.newaxis]
xray_imgs = xray_imgs[..., np.newaxis]

"""

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  
  return result/(num_locations)


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def content_loss(fake_features, real_features, lamda=1e-4):
    loss = 0
    for i in range(len(fake_features)):
        loss += lamda * tl.cost.mean_squared_error(fake_features[i], 
                                                  real_features[i], 
                                                  is_mean=True)        
    return loss


def style_loss(fake_features, real_features, lamda=1e-9):
    loss = 0
    for i in range(len(fake_features)):
        loss += lamda * tl.cost.mean_squared_error(gram_matrix(fake_features[i]), 
                                                  gram_matrix(real_features[i]), 
                                                  is_mean=True)        
    return loss


def perceptual_loss(fake_features, real_features, lamda=1e-4):
    loss = 0
    for i in range(len(fake_features)):
        loss += lamda * tl.cost.absolute_difference_error(fake_features[i], 
                                                          real_features[i], 
                                                          is_mean=True)        
    return loss



def get_train_data_dx(drr_imgs, xray_imgs, size=64):    
    # dataset API and augmentation
    def generator_train():
        for i in range(drr_imgs.shape[0]):
            drr = drr_imgs[i]
            xray = xray_imgs[i]
            #gamma = random.uniform(0.3, 9)
            #xray = gamma_transform(xray, gamma)
            drr = tf.convert_to_tensor(drr)
            xray = tf.convert_to_tensor(xray)
            yield tf.stack([drr, xray], axis = 0)    
    
    def _map_fn_train(img):      
        hr_patch = tf.image.random_crop(img, [2, size, size, 1])
        mr_patch = tf.image.random_crop(img, [2, size, size, 1])
        drr_lr_patch = tf.image.resize(mr_patch[0], size=[size//2, size//2])
        xray_lr_patch = tf.image.resize(mr_patch[1], size=[size//2, size//2]) 
        
        drr_hr_patch = tf.image.resize(mr_patch[0], size=[size, size])
        xray_hr_patch = tf.image.resize(mr_patch[1], size=[size, size])     
        drr_mr_patch = tf.image.resize(mr_patch[0], size=[size, size])
        xray_mr_patch = tf.image.resize(mr_patch[1], size=[size, size])        
        return drr_lr_patch, drr_hr_patch, drr_mr_patch, xray_lr_patch, xray_hr_patch, xray_mr_patch
    
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
        # train_ds = train_ds.repeat(n_epoch_init + n_epoch)
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
        # value = train_ds.make_one_shot_iterator().get_next()
    return train_ds


def get_train_data_sr(drr_imgs, xray_imgs, size=256):    
    # dataset API and augmentation
    def generator_train():
        for i in range(drr_imgs.shape[0]):
            drr = tf.convert_to_tensor(drr_imgs[i])
            xray = tf.convert_to_tensor(xray_imgs[i])
            yield tf.stack([drr, xray], axis = 0)    
    
    def _map_fn_train(img):      
        mr_patch = tf.image.random_crop(img, [2, size, size, 1])
        
        drr_lr_patch = tf.image.resize(mr_patch[0], size=[size//4, size//4])
        xray_lr_patch = tf.image.resize(mr_patch[1], size=[size//4, size//4]) 
        drr_lr_patch = tf.image.resize(drr_lr_patch, size=[size, size])
        xray_lr_patch = tf.image.resize(xray_lr_patch, size=[size, size])  
        drr_hr_patch = mr_patch[0]
        xray_hr_patch = mr_patch[1]               
        return drr_lr_patch, drr_hr_patch, xray_lr_patch, xray_hr_patch
    
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
        # train_ds = train_ds.repeat(n_epoch_init + n_epoch)
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
        # value = train_ds.make_one_shot_iterator().get_next()
    return train_ds


def train():
    G_sr = get_sr_G((batch_size, 256, 256, 1))
    if tl.files.file_exists(os.path.join(checkpoint_dir, 'g_sr_init.h5')):
        G_sr.load_weights(os.path.join(checkpoint_dir, 'g_sr_init.h5'))        
   
    G1 = get_dx_G((batch_size, 256, 256, 1), u_net_blocks=1, name='G11')
    G2 = get_dx_G((batch_size, 256, 256, 1), u_net_blocks=1, name='G2')    
        
    D_sr = get_patch_D((batch_size, 256, 256, 1), name = "sr_discriminator")
    D1 = get_patch_D((batch_size, 256, 256, 1), name = "D11") 
    D2 = get_patch_D((batch_size, 256, 256, 1), name = "D21")  

    
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
    VGG = tl.models.vgg19(pretrained=True, end_with=style_layers, mode='static')

    lr_v = tf.Variable(lr_init)
    g_sr_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
    g1_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
    g2_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)

    g_sr_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    # cycle GAN
    g1_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    g2_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d1_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)  
    d2_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)  
      
    VGG.train()

    size = 256
    train_sr = get_train_data_sr(drr_imgs, xray_imgs, size=size)
    train_dx = get_train_data_dx(drr_imgs, xray_imgs, size=size)

    
"""
    
#---------------------------------INITIALIZE G_DX------------------------------------
    
    ## initialize learning G_dx    
    G1.train() 
    G2.train()
    
    n_step_epoch = round(n_epoch_init)
    for epoch in range(n_epoch_init//5):       
        for step, (_, _, drr_mr_patchs, _, _, xray_mr_patchs) in enumerate(train_dx):           
            if drr_mr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:             
                # generated/fake data              
                fake_xray_patchs = G1(drr_mr_patchs) 
                #cycled_drr_patchs = G2(fake_xray_patchs)
                fake_drr_patchs = G2(xray_mr_patchs)  
                #cycled_xray_patchs = G1(fake_drr_patchs) 
               
                xray_ade_loss = tl.cost.mean_squared_error(fake_xray_patchs, xray_mr_patchs, is_mean=True)
                drr_ade_loss = tl.cost.mean_squared_error(fake_drr_patchs, drr_mr_patchs, is_mean=True)
               
                g1_loss = xray_ade_loss #+ cycle_loss
                g2_loss = drr_ade_loss #+ cycle_loss
            
            
            grad = tape.gradient(g1_loss, G1.trainable_weights)
            g1_optimizer_init.apply_gradients(zip(grad, G1.trainable_weights))
            grad = tape.gradient(g2_loss, G2.trainable_weights)
            g2_optimizer_init.apply_gradients(zip(grad, G2.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g1_ade: {:.5f}, g2_ade: {:.5f}".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time,
                xray_ade_loss,
                drr_ade_loss,
                ))            
        
        tl.vis.save_images(fake_xray_patchs.numpy(), [2, 3], os.path.join(save_dir, 'train_g1_dx_init_{}.png'.format(epoch)))
        tl.vis.save_images(fake_drr_patchs.numpy(), [2, 3], os.path.join(save_dir, 'train_g2_dx_init_{}.png'.format(epoch)))
        G1.save_weights(os.path.join(checkpoint_dir, 'g1_dx_init.h5'))
        G2.save_weights(os.path.join(checkpoint_dir, 'g2_dx_init.h5'))

 
    
    
#---------------------------------INITIALIZE G_SR------------------------------------
           
    ## initialize learning G_sr
    G_sr.train() 
    
    n_step_epoch = round(n_epoch_init // batch_size)
    for epoch in range(n_epoch_init):       
        for step, (drr_lr_patchs, drr_hr_patchs, _, _) in enumerate(train_sr):           
            if drr_lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_drr_hr_patchs = G_sr(drr_lr_patchs)
                #fake_xray_hr_patchs = G_sr(xray_lr_patchs)
                drr_mse_loss_sr = tl.cost.absolute_difference_error(fake_drr_hr_patchs, drr_hr_patchs, is_mean=True)
                #xray_mse_loss_sr = tl.cost.absolute_difference_error(fake_xray_hr_patchs, xray_hr_patchs, is_mean=True)
                mse_loss_sr = drr_mse_loss_sr #+ xray_mse_loss_sr

            grad = tape.gradient(mse_loss_sr, G_sr.trainable_weights)
            g_sr_optimizer_init.apply_gradients(zip(grad, G_sr.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse_sr: {:.5f}".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss_sr))
        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_drr_hr_patchs.numpy(), [1, 2], os.path.join(save_dir, 'train_g_sr_init_{}.png'.format(epoch)))
            G_sr.save_weights(os.path.join(checkpoint_dir, 'g_sr_init.h5'))


#------------------------------ G_SR_____D_SR------------------------------------
    
    ## adversarial learning (G_sr, D_sr)
    if tl.files.file_exists(os.path.join(checkpoint_dir, 'd_sr.h5')):
        D_sr.load_weights(os.path.join(checkpoint_dir, 'd_sr.h5'))
    if tl.files.file_exists(os.path.join(checkpoint_dir, 'g_sr.h5')):
        G_sr.load_weights(os.path.join(checkpoint_dir, 'g_sr.h5'))    
   
    D_sr.train()  
    G_sr.train()

    n_step_epoch = round(n_epoch // batch_size)
    for epoch in range(n_epoch):
        for step, (drr_lr_patchs, drr_hr_patchs, _, _) in enumerate(train_sr):
            if drr_lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                # generated/fake data
                fake_drr_hr_patchs = G_sr(drr_lr_patchs)
                #fake_xray_hr_patchs = G_sr(xray_lr_patchs)
                
                sr_drr_logits_fake = D_sr(fake_drr_hr_patchs)
                #sr_xray_logits_fake = D_sr(fake_xray_hr_patchs)
                sr_drr_feature_fake = VGG((fake_drr_hr_patchs+1)/2.)
                #sr_xray_feature_fake = VGG((fake_xray_hr_patchs+1)/2.)
                # ground-truth/real data
                sr_drr_logits_real = D_sr(drr_hr_patchs)
                #sr_xray_logits_real = D_sr(xray_hr_patchs)
                sr_drr_feature_real = VGG((drr_hr_patchs+1)/2.)
                #sr_xray_feature_real = VGG((xray_hr_patchs+1)/2.)
                
                # D loss  
                d_sr_drr_loss1 = mae_criterion(sr_drr_logits_real[-1], tf.ones_like(sr_drr_logits_real[-1]))
                d_sr_drr_loss2 = mae_criterion(sr_drr_logits_fake[-1], tf.zeros_like(sr_drr_logits_fake[-1]))      
                #d_sr_xray_loss1 = mae_criterion(sr_xray_logits_real[-1], tf.ones_like(sr_xray_logits_real[-1]))
                #d_sr_xray_loss2 = mae_criterion(sr_xray_logits_fake[-1], tf.zeros_like(sr_xray_logits_fake[-1]))      
              
                d_loss = d_sr_drr_loss1 +  d_sr_drr_loss2
                         #+ d_sr_xray_loss1 +  d_sr_xray_loss2)/2.                           
               
                                          
                # G_sr super resolution loss
                g_sr_drr_gan_loss = 1e-2 * mae_criterion(sr_drr_logits_fake[-1], tf.ones_like(sr_drr_logits_fake[-1]))
                #g_sr_xray_gan_loss = 1e-1 * mae_criterion(sr_xray_logits_fake[-1], tf.ones_like(sr_xray_logits_fake[-1]))
                g_sr_drr_ade_loss = tl.cost.absolute_difference_error(fake_drr_hr_patchs, drr_hr_patchs, is_mean=True)
                #g_sr_xray_ade_loss = 10 * tl.cost.absolute_difference_error(fake_xray_hr_patchs, xray_hr_patchs, is_mean=True)
                g_sr_drr_vgg_loss = content_loss(sr_drr_feature_fake, sr_drr_feature_real, lamda=2e-6)
                #g_sr_xray_vgg_loss = content_loss(sr_xray_feature_fake, sr_xray_feature_real, lamda=2e-6)

                g_sr_loss = g_sr_drr_gan_loss\
                            + g_sr_drr_ade_loss\
                            + g_sr_drr_vgg_loss
                                   
            
            grad = tape.gradient(d_loss, D_sr.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D_sr.trainable_weights))           
            grad = tape.gradient(g_sr_loss, G_sr.trainable_weights)
            g_sr_optimizer.apply_gradients(zip(grad, G_sr.trainable_weights))            
            
           
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_sr_loss(ade:{:.5f}, content:{:.5f}, adv:{:.5f}) d_loss:{:.5f}".format(
                epoch, n_epoch, step, n_step_epoch, time.time() - step_time,
                g_sr_drr_ade_loss, 
                g_sr_drr_vgg_loss, 
                g_sr_drr_gan_loss,
                d_loss,
                ))
         
        # update the learning rate
        if epoch > 100 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
            
        
        result = np.concatenate([drr_hr_patchs.numpy(),
                                 fake_drr_hr_patchs.numpy(),
                                 tf.image.resize(drr_lr_patchs, size=[size, size], method="bicubic").numpy()], axis=0)

        
        tl.vis.save_images(result, [3, 2], os.path.join(save_dir, 'train_g_sr_{}.png'.format(epoch)))
        G_sr.save_weights(os.path.join(checkpoint_dir, 'g_sr.h5'))
        D_sr.save_weights(os.path.join(checkpoint_dir, 'd_sr.h5'))
            
       
        
  
#------------------------------ G_DX_____D_DX------------------------------------
               
    ## adversarial learning (G_dx, D_dx)
    if tl.files.file_exists(os.path.join(checkpoint_dir, 'd1_dx.h5')):
        D1.load_weights(os.path.join(checkpoint_dir, 'd1_dx.h5'))
    if tl.files.file_exists(os.path.join(checkpoint_dir, 'd2_dx.h5')):
        D2.load_weights(os.path.join(checkpoint_dir, 'd2_dx.h5'))
    if tl.files.file_exists(os.path.join(checkpoint_dir, 'g1_dx.h5')):
        G1.load_weights(os.path.join(checkpoint_dir, 'g1_dx.h5'))
    if tl.files.file_exists(os.path.join(checkpoint_dir, 'g2_dx.h5')):
        G2.load_weights(os.path.join(checkpoint_dir, 'g2_dx.h5'))
   
    D1.train()
    D2.train()
    G1.train()
    G2.train()
    
    
    n_step_epoch = round(n_epoch // batch_size)
    for epoch in range(n_epoch):
        buffer_d1_fake = []
        buffer_d1_real = []
        buffer_d2_fake = []
        buffer_d2_real = []
        for step, (_, _, drr_mr_patchs, _, _, xray_mr_patchs) in enumerate(train_dx):
            if drr_mr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            
            with tf.GradientTape(persistent=True) as tape:
                # generated/fake data              
                fake_xray_patchs = G1(drr_mr_patchs) 
                cycled_drr_patchs = G2(fake_xray_patchs)               
                fake_drr_patchs = G2(xray_mr_patchs) 
                cycled_xray_patchs = G1(fake_drr_patchs)               
                               
                
                vgg_fake_xray_logits = D1(fake_xray_patchs)
                vgg_fake_drr_logits = D2(fake_drr_patchs)   
                fake_xray_features = VGG((fake_xray_patchs+1)/2.)
                fake_drr_features = VGG((fake_drr_patchs+1)/2.)
                
                
                vgg_real_xray_logits = D1(xray_mr_patchs)
                vgg_real_drr_logits = D2(drr_mr_patchs)
                real_xray_features = VGG((xray_mr_patchs+1)/2.)
                real_drr_features = VGG((drr_mr_patchs+1)/2.)
                
                # cycle loss
                cycle_drr_loss = 10 * tl.cost.absolute_difference_error(cycled_drr_patchs, drr_mr_patchs, is_mean=True)
                cycle_xray_loss = 10 * tl.cost.absolute_difference_error(cycled_xray_patchs, xray_mr_patchs, is_mean=True)
                cycle_loss = cycle_drr_loss + cycle_xray_loss
                
              
                # D1 loss              
                d1_xray_loss1 = mae_criterion(vgg_real_xray_logits[-1:], tf.ones_like(vgg_real_xray_logits[-1:]))
                d1_xray_loss2 = mae_criterion(vgg_fake_xray_logits[-1:], tf.zeros_like(vgg_fake_xray_logits[-1:]))
                d1_loss = (d1_xray_loss1 + d1_xray_loss2)/2.
                         
                # D2 loss
                d2_drr_loss1 = mae_criterion(vgg_real_drr_logits[-1:], tf.ones_like(vgg_real_drr_logits[-1:]))
                d2_drr_loss2 = mae_criterion(vgg_fake_drr_logits[-1:], tf.zeros_like(vgg_fake_drr_logits[-1:]))
                d2_loss = (d2_drr_loss1 + d2_drr_loss2)/2.
                   
                
                # G1 loss
                g1_gan_loss = mae_criterion(vgg_fake_xray_logits[-1:], tf.ones_like(vgg_fake_xray_logits[-1:]))
                g1_xray_ade_loss = tl.cost.absolute_difference_error(fake_xray_patchs, xray_mr_patchs, is_mean=True)
                g1_content_vgg_loss = content_loss(fake_xray_features, real_xray_features, lamda=1e-6)
                g1_style_vgg_loss = style_loss(fake_xray_features, real_xray_features, lamda=1e-10)
                g1_perceptual_vgg_loss = perceptual_loss(vgg_fake_xray_logits[:-1], vgg_real_xray_logits[:-1], lamda=1e-1)
                g1_loss = g1_gan_loss\
                          + cycle_loss\
                          + g1_xray_ade_loss\
                          + g1_perceptual_vgg_loss\
                          + g1_style_vgg_loss\
                          + g1_content_vgg_loss\
                          
                # G2 loss
                g2_gan_loss = mae_criterion(vgg_fake_drr_logits[-1:], tf.ones_like(vgg_fake_drr_logits[-1:]))
                g2_drr_ade_loss = tl.cost.absolute_difference_error(fake_drr_patchs, drr_mr_patchs, is_mean=True)
                g2_content_vgg_loss = content_loss(fake_drr_features, real_drr_features, lamda=1e-6)
                g2_style_vgg_loss = style_loss(fake_drr_features, real_drr_features, lamda=1e-10)
                g2_perceptual_vgg_loss = perceptual_loss(vgg_fake_drr_logits[:-1], vgg_real_drr_logits[:-1], lamda=1e-1)
                g2_loss = g2_gan_loss\
                          + cycle_loss\
                          + g2_drr_ade_loss\
                          + g2_perceptual_vgg_loss\
                          + g2_style_vgg_loss\
                          + g2_content_vgg_loss\
                
                        
           
            grad = tape.gradient(g1_loss, G1.trainable_weights)
            g1_optimizer.apply_gradients(zip(grad, G1.trainable_weights))
            grad = tape.gradient(g2_loss, G2.trainable_weights)
            g2_optimizer.apply_gradients(zip(grad, G2.trainable_weights))            
            if step%1 == 0:
                grad = tape.gradient(d1_loss, D1.trainable_weights)
                d1_optimizer.apply_gradients(zip(grad, D1.trainable_weights))
                grad = tape.gradient(d2_loss, D2.trainable_weights)
                d2_optimizer.apply_gradients(zip(grad, D2.trainable_weights))            
                        
           
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g1_loss(cycle:{:.5f}, ade:{:.5f}, content:{:.5f}, style:{:.5f}, perceptual:{:.5f}, adv:{:.5f}) d1_loss:{:.5f}".format(
                epoch, n_epoch, step, n_step_epoch, time.time() - step_time,               
                 cycle_drr_loss, 
                 g1_xray_ade_loss,
                 g1_content_vgg_loss,
                 g1_style_vgg_loss,
                 g1_perceptual_vgg_loss,
                 g1_gan_loss, 
                 d1_loss))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g2_loss(cycle:{:.5f}, ade:{:.5f}, content:{:.5f}, style:{:.5f}, perceptual:{:.5f}, adv:{:.5f}) d2_loss:{:.5f}".format(
                epoch, n_epoch, step, n_step_epoch, time.time() - step_time,               
                 cycle_xray_loss, 
                 g2_drr_ade_loss,
                 g2_content_vgg_loss,
                 g2_style_vgg_loss,
                 g2_perceptual_vgg_loss,
                 g2_gan_loss,
                 d2_loss))

        # update the learning rate
        if epoch > 100 and (epoch % (decay_every//2) == 0):
            new_lr_decay = lr_decay**(epoch // (decay_every//2))
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch % 1 == 0):
            tl.vis.save_images(fake_xray_patchs.numpy(), [2, 2], os.path.join(save_dir, 'train_g1_xrays_{}.png'.format(epoch)))
            tl.vis.save_images(fake_drr_patchs.numpy(), [2, 2], os.path.join(save_dir, 'train_g2_drrs_{}.png'.format(epoch)))
            G1.save_weights(os.path.join(checkpoint_dir, 'g11_dx.h5'))
            D1.save_weights(os.path.join(checkpoint_dir, 'd11_dx.h5'))
            G2.save_weights(os.path.join(checkpoint_dir, 'g21_dx.h5'))
            D2.save_weights(os.path.join(checkpoint_dir, 'd21_dx.h5'))
            
            
            
            
            
            
#------------------------------ G_SR_____G_DX_____D_DX------------------------------------
           
    ## adversarial learning (G_dx, D_dx)
    if tl.files.file_exists(os.path.join(checkpoint_dir, 'd1_dx.h5')):
        D1.load_weights(os.path.join(checkpoint_dir, 'd1_dx.h5'))
    if tl.files.file_exists(os.path.join(checkpoint_dir, 'g1_dx.h5')):
        G1.load_weights(os.path.join(checkpoint_dir, 'g1_dx.h5'))
    if tl.files.file_exists(os.path.join(checkpoint_dir, 'g_sr.h5')):
        G_sr.load_weights(os.path.join(checkpoint_dir, 'g_sr.h5'))
   
    D1.train()  
    G1.train() 
    G_sr.eval()
    
    
    n_step_epoch = round(n_epoch // batch_size)
    for epoch in range(n_epoch):
        for step, (drr_lr_patchs, drr_hr_patchs, _, _, xray_hr_patchs, _) in enumerate(train_dx):
            if drr_lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                # generated/fake data     
                fake_sr_drr_patchs = G_sr(drr_lr_patchs)                       
                fake_sr_xray_patchs = G1(fake_sr_drr_patchs)
                fake_hr_xray_patchs = G1(drr_hr_patchs)
                
                fake_sr_xray_logits = D1(fake_sr_xray_patchs)
                fake_hr_xray_logits = D1(fake_hr_xray_patchs)
                fake_sr_xray_features = VGG((fake_sr_xray_patchs+1)/2.)
                fake_hr_xray_features = VGG((fake_hr_xray_patchs+1)/2.)
                # ground-truth/real data
                real_xray_logits = D1(xray_hr_patchs)
                real_xray_features = VGG((xray_hr_patchs+1)/2.)
                
                # D loss                
                d1_dx_loss1 = mae_criterion(real_xray_logits[-1:], tf.ones_like(real_xray_logits[-1:]))
                d1_dx_loss2 = mae_criterion(fake_sr_xray_logits[-1:], tf.zeros_like(fake_sr_xray_logits[-1:]))
                d1_dx_loss3 = mae_criterion(fake_hr_xray_logits[-1:], tf.zeros_like(fake_hr_xray_logits[-1:]))
                d_loss = (2*d1_dx_loss1 + d1_dx_loss2 + d1_dx_loss3)/4.        
                
                 # G1 loss
                g1_gan_loss1 = mae_criterion(fake_sr_xray_logits[-1:], tf.ones_like(fake_sr_xray_logits[-1:]))
                g1_gan_loss2 = mae_criterion(fake_hr_xray_logits[-1:], tf.ones_like(fake_hr_xray_logits[-1:]))
                g1_xray_ade_loss1 = 1e-1 * tl.cost.absolute_difference_error(fake_sr_xray_patchs, xray_hr_patchs, is_mean=True)
                g1_xray_ade_loss2 = 1e-1 * tl.cost.absolute_difference_error(fake_hr_xray_patchs, xray_hr_patchs, is_mean=True)
                g1_content_vgg_loss1 = content_loss(fake_sr_xray_features, real_xray_features, lamda=1e-6)
                g1_content_vgg_loss2 = content_loss(fake_hr_xray_features, real_xray_features, lamda=1e-6)
                g1_style_vgg_loss1 = style_loss(fake_sr_xray_features, real_xray_features, lamda=1e-10)
                g1_style_vgg_loss2 = style_loss(fake_hr_xray_features, real_xray_features, lamda=1e-10)
                g1_perceptual_vgg_loss1 = perceptual_loss(fake_sr_xray_logits[:-1], real_xray_logits[:-1], lamda=1e-2)
                g1_perceptual_vgg_loss2 = perceptual_loss(fake_hr_xray_logits[:-1], real_xray_logits[:-1], lamda=1e-2)
                g1_loss = g1_gan_loss1 + g1_gan_loss2\
                          + g1_xray_ade_loss1 + g1_xray_ade_loss2\
                          + g1_style_vgg_loss1 + g1_style_vgg_loss2\
                          + g1_perceptual_vgg_loss1 + g1_perceptual_vgg_loss2\
                          + g1_content_vgg_loss1 + g1_content_vgg_loss2\
                        
           
            grad = tape.gradient(g1_loss, G1.trainable_weights)
            g1_optimizer.apply_gradients(zip(grad, G1.trainable_weights))            
            grad = tape.gradient(d_loss, D1.trainable_weights)
            d1_optimizer.apply_gradients(zip(grad, D1.trainable_weights))            
                    
           
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_dx_loss(ade:{:.5f}, content:{:.5f}, style:{:.5f}, perceptual:{:.5f}, adv_1:{:.5f}, adv_2:{:.5f}) d_loss: {:.5f}".format(
                epoch, n_epoch, step, n_step_epoch, time.time() - step_time, 
                g1_xray_ade_loss1 + g1_xray_ade_loss2, 
                g1_content_vgg_loss1 + g1_content_vgg_loss2,
                g1_style_vgg_loss1 + g1_style_vgg_loss2,
                g1_perceptual_vgg_loss1 + g1_perceptual_vgg_loss2,
                g1_gan_loss1,
                g1_gan_loss2,
                d_loss))
            
        result = np.concatenate([xray_hr_patchs.numpy(),
                                 fake_sr_xray_patchs.numpy(),
                                 fake_hr_xray_patchs.numpy()
                                 ], axis=0)
        

        # update the learning rate
        if epoch > 100 and (epoch % (decay_every) == 0):
            new_lr_decay = lr_decay**(epoch // (decay_every))
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        
        tl.vis.save_images(result, [3, 3], os.path.join(save_dir, 'train_g_sr_dx_{}.png'.format(epoch)))
        G1.save_weights(os.path.join(checkpoint_dir, 'g1_dx.h5'))
        D1.save_weights(os.path.join(checkpoint_dir, 'd1_dx.h5'))
            
            
            
    
    
         
            
    

def evaluate():
    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## if your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)

    ###========================== DEFINE MODEL ============================###
    imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    G_sr = get_sr_G([1, None, None, 1])
    G1 = get_dx_G([1, 512, 512, 1], u_net_blocks = 1, refine=False, name = 'g1')
    G2 = get_dx_G([1, 256, 256, 1], u_net_blocks = 1, refine=False, name = 'g12')

    
    G_sr.load_weights(os.path.join(checkpoint_dir, 'g_sr.h5'))
    G1.load_weights(os.path.join(checkpoint_dir, 'g1_dx.h5'))
    G2.load_weights(os.path.join(checkpoint_dir, 'g1_dx.h5'))
    G_sr.eval()
    G1.eval()
    G2.eval()

    
  

    valid_drr_lr_img = scipy.ndimage.interpolation.zoom(imgs[250, :, 128:896], 
                                                    zoom = 0.333)[..., np.newaxis]    
    valid_drr_lr_img = valid_drr_lr_img[np.newaxis, ...].astype('float32')
    valid_xray_lr_img = scipy.ndimage.interpolation.zoom(xray_imgs[90, :, 128:896, 0], 
                                                    zoom = 0.25)[..., np.newaxis]
    valid_xray_lr_img = valid_xray_lr_img[np.newaxis, ...].astype('float32')

    fake_lr_xray = G2(valid_drr_lr_img).numpy()

    fake_sr_drr = G_sr(imgs_resized[90:91][..., np.newaxis]).numpy()
    fake_sr_xray = G_sr(valid_xray_lr_img).numpy()
    
    fake_lr_xray = G1(fake_sr_drr).numpy()
    fake_lr_drr = G2(fake_sr_xray).numpy()
    
    fake_hr_xray = G1(drr_imgs[135:136, :-128, 128:-128, :]).numpy()
    fake_hr_drr = G2(xray_imgs[90:91, :, 128:896, :]).numpy()
    

       
    
    plt.imshow(fake_hr_xray[0,...,0], cmap = 'gray', vmax=-0.5)
    plt.figure()
    plt.imshow(xray_imgs[135, :-128, 128:-128, 0], cmap = 'gray', vmax=-0.5)
    plt.figure()
    plt.imshow(drr_imgs[135, :-128, 128:-128, 0], cmap = 'gray', vmax=-0.5)
    plt.figure()
    plt.imshow(fake_xray[0, ..., 0], cmap = 'gray')
    plt.figure()
    plt.imshow(drr_hr_patchs[0, ..., 0], cmap = 'gray')
    plt.figure()
    plt.imshow(xray_hr_patchs[0, ..., 0], cmap = 'gray')

    plt.imshow(drr_imgs[90, ..., 0], cmap = 'gray')
    plt.figure()
    plt.imshow(fake_hr_xray[0, ..., 0], cmap = 'gray')
    plt.figure()
    plt.imshow(fake_hr_drr[0, ..., 0], cmap = 'gray')
    plt.figure()
    plt.imshow(xray_imgs[90, ..., 0], cmap = 'gray')
    
    plt.imshow(fake_sr_xray[0,...,0], cmap = 'gray')
    plt.figure()
    plt.imshow(fake_sr_drr[0, ..., 0], cmap = 'gray')
    plt.figure()
    plt.imshow(imgs[250], cmap = 'gray')
    plt.figure()    
    plt.imshow(fake_lr_xray[0, ..., 0], cmap = 'gray', vmax=0.3)
    plt.figure()
    plt.imshow(fake_lr_drr[0, ..., 0], cmap = 'gray')
    


    
    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], os.path.join(save_dir, 'valid_gen.png'))
    tl.vis.save_image(valid_lr_img[0], os.path.join(save_dir, 'valid_lr.png'))
    tl.vis.save_image(valid_hr_img, os.path.join(save_dir, 'valid_hr.png'))

    out_bicu = scipy.misc.imresize(valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, os.path.join(save_dir, 'valid_bicubic.png'))

"""

