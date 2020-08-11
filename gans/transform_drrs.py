#! /usr/bin/python
# -*- coding: utf8 -*-
import sys
sys.path.append('D:/MasterAIThesis/code/load_dicom/')
sys.path.append('D:/MasterAIThesis/code/srgan/')
sys.path.append('D:/MasterAIThesis/code/srgan/models/')
sys.path.append('D:/MasterAIThesis/code/')

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
import index_tracker as track   



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def transform_drrs(checkpoint_dir, drr_path, patient_name, model = 'mrcnn', data_mode = 'shifted'):
     assert data_mode in ['standard', 'decorrelated']
     assert model in ['frcnn', 'mrcnn', 'siammask']
     
     load_path = drr_path + patient_name + '/DRRs/' + data_mode + '/'
     
     G_sr = get_sr_G([1, None, None, 1])
     G_dx = get_dx_G([1, 512, 512, 1], u_net_blocks = 1, refine=False, name = 'g1')
    
     G_sr.load_weights(os.path.join(checkpoint_dir, 'g_sr.h5'))
     G_dx.load_weights(os.path.join(checkpoint_dir, 'g1_dx.h5'))
     G_sr.eval()
     G_dx.eval()
     
     for i in range(0, 1):   
         # load data
         drr_dir = load_path + str(10*i) + '-' + str(10*(i+1)) + '/' 
         imgs_name = data_mode + '_drr_imgs'
         imgs = dtn.load_data(drr_dir + imgs_name + '.hdf5', imgs_name)
              
         c = 1/np.log(1 + np.max(imgs, axis = (1,2)))
         imgs = np.log(imgs+1)
         imgs = np.multiply(c[..., np.newaxis, np.newaxis], imgs)
         imgs = (imgs - np.min(imgs))/(np.max(imgs)-np.min(imgs))
         imgs = imgs*2 - 1
         
         imgs = imgs[:, 128:-128, 256:-256]         

         
         imgs_srgan = []
         imgs_cygan = []
         imgs_srcygan = []
         for j in range(imgs.shape[0]):
             print(j)
             sr_img = G_sr(imgs[j:j+1][..., np.newaxis]).numpy()
             imgs_srgan.append(sr_img[..., 0])
         imgs_srgan = np.concatenate(imgs_srgan, axis = 0)

         for j in range(imgs.shape[0]):
             print(j)   
             dx_img = G_dx(imgs[j:j+1][..., np.newaxis]).numpy()
             sr_dx_img = G_dx(imgs_srgan[j:j+1][..., np.newaxis]).numpy()
             imgs_cygan.append(dx_img[..., 0])
             imgs_srcygan.append(sr_dx_img[..., 0]) 
             
         imgs_cygan = np.concatenate(imgs_cygan, axis = 0)        
         imgs_srcygan = np.concatenate(imgs_srcygan, axis = 0)
         
         dtn.save_data(drr_dir, imgs_name + '_srgans', imgs_srgan)
         dtn.save_data(drr_dir, imgs_name + '_cygans', imgs_cygan)
         dtn.save_data(drr_dir, imgs_name + '_srcygans', imgs_srcygan)
             
         print(i)
  
"""
checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)
drr_path = 'D:/MasterAIThesis/h5py_data/vumc phantom data/'
patient_name = 'phantom3'
model = 'siammask'
data_mode = 'decorrelated'


transform_drrs(checkpoint_dir, drr_path, patient_name, model, data_mode)


final_imgs = np.stack([imgs, imgs_gans], axis = 0)        

imgs = imgs[np.newaxis, ...]
imgs_srgan = imgs_srgan[np.newaxis, ...]
imgs_cygan = imgs_cygan[np.newaxis, ...]
imgs_srcygan = imgs_srcygan[np.newaxis, ...]

X = np.transpose(imgs_srcygan, (0, 2, 3, 1))
Z = Y = np.zeros(X.shape, dtype = 'uint8')

fig, ax = plt.subplots(1, 1)
extent = (0, X.shape[2], 0, X.shape[1])
tracker1 = track.IndexTracker(ax, X, Y, Z, extent, vmin = -1, vmax = 0.1)

fig.canvas.mpl_connect('key_press_event', tracker1.onpress)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress2)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress3)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress4)
fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)

"""



