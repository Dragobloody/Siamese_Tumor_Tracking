import sys
sys.path.append('D:/MasterAIThesis/code')
sys.path.append('D:/MasterAIThesis/code/load_dicom/')
sys.path.append('D:/MasterAIThesis/code/load_dicom/MedImgSeg')

import dicom_to_numpy as dtn
import prepare_patient_data as ppd
import numpy as np
import matplotlib.pyplot as plt
from MedImgSeg.preprocess  import normalize_set_3D, window_set_HU
from MedImgSeg.dicom import load_dataset

import index_tracker as track   


# insert your data directory
data_directory = 'D:/MasterAIThesis/dicom_data/vumc phantom data/'
# discover names for structure in the dataset
roi_tags = [['ITV3'], ['BODY']]
not_roi_tags = [[], []]

# GET DATASET
p_dict = ppd.prepare_p_dict(data_directory, roi_tags, not_roi_tags)
h5py_data_directory = 'D:/MasterAIThesis/h5py_data/vumc phantom data/'
dtn.create_cases(p_dict, h5py_data_directory)

base_name = "phase_"
dataset = load_dataset(h5py_data_directory, base_name)

save_dir = 'D:/MasterAIThesis/h5py_data/vumc phantom data/'
ppd.generate_patient_data(save_dir, dataset)



# LOAD DATA
load_data_path = 'D:/MasterAIThesis/h5py_data/vumc phantom data/phantom3/original_data/'
imgs = dtn.load_data(load_data_path + 'imgs.hdf5', 'imgs')
labs = dtn.load_data(load_data_path + 'labs.hdf5', 'labs')

save_data_path = 'D:/MasterAIThesis/h5py_data/vumc phantom data/phantom3/deformed_data/0-10/'
dtn.save_data(save_data_path, 'deformed_imgs', imgs)
dtn.save_data(save_data_path, 'deformed_labs', labs)


# LOAD MODEL

deformed_imgs = dtn.load_data(load_data_path + 'deformed_imgs.hdf5', 'deformed_imgs')
deformed_labs = dtn.load_data(load_data_path + 'deformed_labs.hdf5', 'deformed_labs')

shifted_imgs = dtn.load_data(load_data_path + 'shifted_imgs.hdf5', 'shifted_imgs')
shifted_labs = dtn.load_data(load_data_path + 'shifted_labs.hdf5', 'shifted_labs')

#------------------------------------------------------------------------

X = np.transpose(imgs, (0, 2, 3, 1))
Z = Y = np.transpose(labs, (0, 2, 3, 1))

fig, ax = plt.subplots(1, 1)
extent = (0, X.shape[3], 0, X.shape[1])
tracker1 = track.IndexTracker(ax, X, Y, Z, extent, vmin = -1000, vmax =1000)

fig.canvas.mpl_connect('key_press_event', tracker1.onpress)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress2)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress3)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress4)
fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)   
   

