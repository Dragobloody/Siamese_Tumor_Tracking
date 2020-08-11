from scipy import misc
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
from MaskRCNN import visualize

    
    
image_path_mass = glob.glob('C:/Users/DRAGOS/Desktop/sample/images/00010815_006.png')  
image_path_nodule = glob.glob('C:/Users/DRAGOS/Desktop/sample/images/00013951_001.png')  
image_mass = imageio.imread(image_path_mass[0]).astype('float32')
image_nodule = imageio.imread(image_path_nodule[0])
   



x, y = 311.182222222222, 241.531267361111
w, h = 146.773333333333, 256
x1 = x 
x2 = x1 + w
y1 = y 
y2 = y1 + h

bbox_mass = np.array([y1, x1, y2, x2])


x, y = 301.240211640212, 266.565079365079
w, h = 62.8486772486773, 49.8455026455026
x1 = x 
x2 = x1 + w
y1 = y 
y2 = y1 + h

bbox_nodule = np.array([y1, x1, y2, x2])


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


ax = get_ax(1)
visualize.display_instances(image_mass, bbox_mass[np.newaxis, :], np.zeros((1024, 1024, 1)), np.array([1]), 
                            ['bg',''], ax=ax, show_mask = False,
                            title="Predictions")

ax = get_ax(1)
visualize.display_instances(image_nodule, bbox_nodule[np.newaxis, :], np.zeros((1024, 1024, 1)), np.array([1]), 
                            ['bg',''], ax=ax, show_mask = False,
                            title="Predictions")


import cv2 as cv
image_mass = cv.normalize(image_mass, image_mass, 0, 1, cv.NORM_MINMAX)
plt.imshow(image_mass, cmap = 'gray')
