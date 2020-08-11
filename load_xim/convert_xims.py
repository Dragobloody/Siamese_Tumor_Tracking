import numpy as np
import h5py
import glob
from matplotlib import pyplot as plt
from load_xim.ximmer2 import XimReader
import pydicom

path = "D:/tracking 2 ro/"
patient = "20200402173942 arc1_4"
arc_folder = path + patient + '/'
save_jpgs = True
save_h5 = True

arc_folders = glob.glob(arc_folder)
for arc_folder in arc_folders:
    file_list = sorted(glob.glob(arc_folder + "/*.xim"))
    xim = XimReader(file_list[0])
    xim.headerData()
    dimensions = (len(file_list), xim.ximHeader['Height'], xim.ximHeader['Width'])
    image = np.zeros(dimensions)
    rotations = []
    for i in range(len(file_list)):
        xim = XimReader(file_list[i])
        xim.headerData()
        xim.pixelData()
        xim.histogramData()
        xim.propertiesData()
        image[i,:,:] = xim.uncompressedImage
        rotations.append(xim.propertyDict['KVSourceRtn'])
        print(str(i) + ' of ' + str(len(file_list)))
        
    image = image.astype('float32')
    if(save_jpgs):
        for i in range(0,image.shape[0]):
            plt.imsave(file_list[i][:-4] + ".jpg", image[i,:,:], cmap="gray")

    if(save_h5):
        with h5py.File(arc_folder + "/" + arc_folder.split('/')[-1] + patient + ".h5", 'w') as f:
            f.create_dataset('img', data=image, compression='gzip')
            f.create_dataset('rot', data=rotations, compression='gzip')


plt.imshow(image[10], cmap = 'gray')
plt.figure()
plt.imshow(image[100], cmap = 'gray', vmax = 1000)

mean = np.mean(image)
std = np.std(image)
image = (image-mean)/std


xray = image.copy()
max_val = 1000
for i in range(xray.shape[0]):   
    xray[i][xray[i] > max_val] = max_val
    xray[i] = - np.log(xray[i] + np.min(xray[i]) + 1) 
    
    

plt.imshow(xray[100], cmap = 'gray')


path_file_1 = "C:/lung tracking phantom data challenge/MATCH upload/Intratreatment images/Plan1 High Complexity Motion Trajectory/2020-01-08 19-14-29-kV Fluoro-120kV/"
path_file_2 = "C:/lung tracking phantom data challenge/MATCH upload/Intratreatment images/Plan2 Mean complexity Trajectory/2020-01-08 21-18-25-kV Fluoro-120kV/"
ct_file = path_file_2 + "00400_masked.dcm"
ct = pydicom.read_file(ct_file)
array = ct.pixel_array        



plt.imshow(array, cmap = 'gray')


