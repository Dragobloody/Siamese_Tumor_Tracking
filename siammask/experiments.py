import sys
sys.path.append('D:/MasterAIThesis/code/load_dicom/')
import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import numpy as np


import index_tracker as track   
import csv
import cv2
import scipy
import h5py
import skimage
import time


import dicom_to_numpy as dtn
import matplotlib.pyplot as plt
import seaborn as sns


def generate_class_scores(model, data_generator, config, steps=10):
    scores = []
    for i in range(steps):
        print(i)
        inputs, _ = next(data_generator)
        outputs = model.keras_model.predict(inputs)
        if config.MASK:
            scores.append(outputs[15])
        else:
            scores.append(outputs[12])
    scores = np.concatenate(scores, axis = 0)
    
    return scores

def get_classification_map(pred_class):
    scores = pred_class[..., -1]
    s = scores.shape
    l = np.sqrt(s[1]).astype('int16')
    scores = np.reshape(scores, (s[0], l, l))
    scores = np.mean(scores, axis = 0)
    
    return scores



all_maps = []
scores = generate_class_scores(model, data_generator_drrs, config, steps=10)
class_map = get_classification_map(scores)
all_maps.append(class_map)
plt.imshow(class_map)
plt.imshow(all_maps[-3], vmin=0, vmax=1)
plt.figure()
plt.imshow(all_maps[-2], vmin=0, vmax=1)
plt.figure()
plt.imshow(all_maps[-1], vmin=0, vmax=1)
plt.figure()
plt.imshow(class_map, vmin=0, vmax=1)




def plot_heatmap(maps):
   
    f,(ax1,ax2,ax3,ax4, ax5, axcb) = plt.subplots(1,len(maps)+1, 
                #sharex = True,
                figsize=(3*(len(maps)+1), 3),
                gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
    g1 = sns.heatmap(maps[0],vmin=0,vmax=1,cmap="jet",cbar=False,ax=ax1)
    g1.set_title('No shifting')
    g1.axis('equal')
    g1.set_xticks([])
    g1.set_yticks([])
    g2 = sns.heatmap(maps[1],vmin=0,vmax=1,cmap="jet",cbar=False,ax=ax2)
    g2.set_title('16-32 shifting')
    g2.axis('equal')
    g2.set_xticks([])
    g2.set_yticks([])
    g3 = sns.heatmap(maps[2],vmin=0,vmax=1,cmap="jet",cbar=False,ax=ax3)
    g3.set_title('32-64 shifting')
    g3.axis('equal')
    g3.set_xticks([])
    g3.set_yticks([])
    g4 = sns.heatmap(maps[3],vmin=0,vmax=1,cmap="jet",cbar=False, ax=ax4)
    g4.set_title('Blank target')
    g4.axis('equal')
    g4.set_xticks([])
    g4.set_yticks([]) 
    g5 = sns.heatmap(maps[4],vmin=0,vmax=1,cmap="jet",ax=ax5, cbar_ax=axcb)
    g5.set_title('Random noise target')
    g5.axis('equal')
    g5.set_xticks([])
    g5.set_yticks([])
    
    plt.show()


