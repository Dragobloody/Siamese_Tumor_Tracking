import sys
sys.path.append('D:/MasterAIThesis/code/load_dicom/')
import os
import dicom_to_numpy as dtn
import h5py
import numpy as np
import matplotlib.pyplot as plt


def save_data(data_path, file_name, data):
    # save data    
    f_name = data_path + file_name + ".hdf5"
    f = h5py.File(f_name)
    f.create_dataset(file_name, data = data)
    f.close()


def load_data(data_path, data_name):
    # load data    
    f = h5py.File(data_path,"r")
    data = f[data_name][:]
    f.close()
    return data

xrays, drrs = [], []

drr_angles = np.arange(91, 451, 1)
data_path = 'D:/MasterAIThesis/h5py_data/vumc patient data/'
patient_name = 'patient1'
xray_data_path = data_path + patient_name + '/X_rays/'

# LOAD XRAYS
xrays1 = dtn.load_data(xray_data_path + 'tumor2.h5', 'img')
xrays1 = xrays1.astype('float32')
angles1 = dtn.load_data(xray_data_path + 'patient1_x_rays.h5', 'rot')
xrays2 = dtn.load_data(xray_data_path + 'tumor1.h5', 'img')
xrays2 = xrays2.astype('float32')
angles2 = dtn.load_data(xray_data_path + 'tumor1.h5', 'rot')

# LOAD DRRS
drr_data_path = data_path + patient_name + '/DRRs/'
drrs1 = load_data(drr_data_path + 'standard/0-10/' + 'standard_drr_imgs.hdf5','standard_drr_imgs')
drrs2 = load_data(drr_data_path + 'standard/10-20/' + 'standard_drr_imgs.hdf5','standard_drr_imgs')

indices = []
for a in drr_angles:
    diff = abs(angles1 - a)
    idx = np.argmin(diff)
    indices.append(idx)
    
xrays1 = xrays1[indices]
xrays.append(xrays1)
drrs.append(drrs1)


indices = []
for a in drr_angles:
    diff = abs(angles2 - a)
    idx = np.argmin(diff)
    indices.append(idx)
    
xrays2 = xrays2[indices]
xrays.append(xrays2)
drrs.append(drrs2)

xrays = np.concatenate(xrays, axis = 0)
drrs = np.concatenate(drrs, axis = 0)




mean = np.mean(xrays)
std = np.std(xrays)
xrays = (xrays - mean) / std  
xrays = (xrays - np.min(xrays))/(np.max(xrays) - np.min(xrays))
xrays = xrays*2 - 1
# data normalization
mean = np.mean(drrs)
std = np.std(drrs)
drrs = (drrs - mean) / std  
drrs = (drrs - np.min(drrs))/(np.max(drrs) - np.min(drrs))
drrs = drrs*2 - 1

save_dir = 'D:/MasterAIThesis/code/srgan/dataset/'
save_data(save_dir, 'drrs', drrs)
save_data(save_dir, 'xrays', xrays)



plt.imshow(xray_imgs[90], cmap = 'gray', vmax = 0)
plt.figure()
plt.imshow(drr_imgs[90], cmap = 'gray', vmax = 0)

