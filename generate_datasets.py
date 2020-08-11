import sys
sys.path.append('D:/MasterAIThesis/code/siammask/')

import numpy as np
import matplotlib.pyplot as plt
import h5py
import index_tracker as track   
import utils

def load_data(data_path, data_name):
    # load data    
    f = h5py.File(data_path,"r")
    data = f[data_name][:]
    f.close()
    return data

def save_data(data_path, file_name, data):
    # save data    
    f_name = data_path + file_name + ".hdf5"
    f = h5py.File(f_name)
    f.create_dataset(file_name, data = data)
    f.close()
    

def crop_around_center(data, crop_x, crop_y):
  data = data[:, crop_x//2:-crop_x//2, crop_y//2:-crop_y//2]

  return data




def generate_test_dataset(drr_path, patient_name, model, data_mode, gans=None):
    assert model in ['frcnn', 'mrcnn', 'siammask']

    load_path = drr_path + patient_name + '/DRRs/' + data_mode + '/'
    test_imgs, test_labs = [], []

    for i in range(8, 10):   
         # load data
         drr_dir = load_path + str(10*i) + '-' + str(10*(i+1)) + '/' 
         imgs_name = data_mode + '_drr_imgs'
         labs_name = data_mode + '_drr_labs'  
         if gans is not None:
             imgs_name = imgs_name + '_' + gans
             
         imgs = load_data(drr_dir + imgs_name + '.hdf5', imgs_name)
         labs = load_data(drr_dir + labs_name + '.hdf5', labs_name)
         
         # apply log transformation to make data more linear
         # and normalize between -1 and 1
         if gans is None:
             imgs = utils.log_transform(imgs)
         
         # reshape data 
         imgs = np.reshape(imgs, (imgs.shape[0] // 181, 181, imgs.shape[1], imgs.shape[2]))
         labs = np.reshape(labs, (labs.shape[0] // 181, 181, labs.shape[1], labs.shape[2]))                 
                   
         if model in ['frcnn', 'mrcnn']:
                 
             imgs = np.moveaxis(imgs, 1, 0)
             labs = np.moveaxis(labs, 1, 0)         
             imgs = np.reshape(imgs[:-1, ...], (9, 20, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
             labs = np.reshape(labs[:-1, ...], (9, 20, labs.shape[1], labs.shape[2], labs.shape[3]))         
             imgs = np.reshape(imgs, (imgs.shape[0], imgs.shape[1]*imgs.shape[2], imgs.shape[3], imgs.shape[4]))
             labs = np.reshape(labs, (labs.shape[0], labs.shape[1]*labs.shape[2], labs.shape[3], labs.shape[4]))
         
             # crop around center
             imgs = imgs[:, :, imgs.shape[2]//4:-imgs.shape[2]//4, imgs.shape[3]//4:-imgs.shape[3]//4]
             labs = labs[:, :, labs.shape[2]//4:-labs.shape[2]//4, labs.shape[3]//4:-labs.shape[3]//4]
                          
             
         elif model in ['siammask']:            
             # crop around center
             if gans is None:
                 imgs = imgs[:, :, imgs.shape[2]//4:-imgs.shape[2]//4, imgs.shape[3]//4:-imgs.shape[3]//4]
             else: 
                 imgs = imgs[:, :, 64:-64, :]

             labs = labs[:, :, labs.shape[2]//4:-labs.shape[2]//4, labs.shape[3]//4:-labs.shape[3]//4]
                         
         else:
             raise NotImplementedError
             
         test_imgs.append(imgs)
         test_labs.append(labs)
             
         print(i)             
     
    test_imgs = np.concatenate(test_imgs, axis = 0)
    test_labs = np.concatenate(test_labs, axis = 0)
    
    save_path = drr_path + patient_name+ '/models/' + model + '/' + data_mode + '/dataset/'
    if gans is not None:
        save_data(save_path, 'test_imgs'+ '_' + gans, test_imgs)
    else:
        save_data(save_path, 'test_imgs', test_imgs)        
        save_data(save_path, 'test_labs', test_labs)
     

def generate_train_dataset(drr_path, patient_name, model = 'mrcnn', data_mode = 'shifted', gans=None):
     assert data_mode in ['standard', 'shifted', 'decorrelated']
     assert model in ['frcnn', 'mrcnn', 'siammask']
     
     load_path = drr_path + patient_name + '/DRRs/' + data_mode + '/'
     
     train_imgs, train_labs = [], []     
     for i in range(0, 8):   
         # load data
         drr_dir = load_path + str(10*i) + '-' + str(10*(i+1)) + '/' 
         imgs_name = data_mode + '_drr_imgs'
         labs_name = data_mode + '_drr_labs'   
         if gans is not None:
             imgs_name = imgs_name + '_' + gans
             
         imgs = load_data(drr_dir + imgs_name + '.hdf5', imgs_name)
         labs = load_data(drr_dir + labs_name + '.hdf5', labs_name)    
         
         # apply log transformation to make data more linear
         # and normalize between -1 and 1
         if gans is None:
             imgs = utils.log_transform(imgs)
          
         imgs = np.reshape(imgs, (imgs.shape[0] // 181, 181, imgs.shape[1], imgs.shape[2]))
         labs = np.reshape(labs, (labs.shape[0] // 181, 181, labs.shape[1], labs.shape[2]))
         
         if model in ['frcnn', 'mrcnn']:            
             imgs = imgs[:, 0::5, :, :]
             labs = labs[:, 0::5, :, :]
             
             imgs = np.reshape(imgs, (imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3]))
             labs = np.reshape(labs, (labs.shape[0]*labs.shape[1], labs.shape[2], labs.shape[3]))
                         
             # crop around center
             imgs = crop_around_center(imgs, imgs.shape[1]//2, imgs.shape[2]//2)
             labs = crop_around_center(labs, labs.shape[1]//2, labs.shape[2]//2)                 
             
        
         elif model in ['siammask']:         
             # crop around center
             if gans is None:
                 imgs = imgs[:, :, imgs.shape[2]//4:-imgs.shape[2]//4, imgs.shape[3]//4:-imgs.shape[3]//4]
             else: 
                 imgs = imgs[:, :, 64:-64, :]

             labs = labs[:, :, labs.shape[2]//4:-labs.shape[2]//4, labs.shape[3]//4:-labs.shape[3]//4]
                         
         else:
             raise NotImplementedError
              
         train_imgs.append(imgs)
         train_labs.append(labs)       
             
         print(i)
             
     train_imgs = np.concatenate(train_imgs, axis = 0)
     train_labs = np.concatenate(train_labs, axis = 0)     
     
     save_path = drr_path + patient_name+ '/models/' + model + '/' + data_mode + '/dataset/'
     if gans is not None:
        save_data(save_path, 'train_imgs'+ '_' + gans, train_imgs)
     else:
        save_data(save_path, 'train_imgs', train_imgs)        
        save_data(save_path, 'train_labs', train_labs)
    


drr_path = 'D:/MasterAIThesis/h5py_data/vumc patient data/'
patient_name = 'patient1'
model = 'siammask'
data_mode = 'decorrelated'

gans = 'cygans'
gans = None

generate_train_dataset(drr_path, patient_name, model, data_mode, gans=gans)
generate_test_dataset(drr_path, patient_name, model, data_mode, gans=gans)



save_path = drr_path + patient_name+ '/models/' + model + '/' + data_mode + '/dataset/'
i= 0
load_dir = drr_path + patient_name + '/deformed_data/' + str(10*i) + '-' + str(10*(i+1)) + '/'

imgs = load_data(load_dir + 'deformed_imgs.hdf5', 'deformed_imgs')
labs = load_data(load_dir + 'deformed_labs.hdf5', 'deformed_labs')




data_path = 'D:/MasterAIThesis/h5py_data/vumc phantom data/'
patient_name = 'phantom2_2'
model = 'siammask'
data_mode = 'standard'
drr_data_path = data_path + patient_name + '/models/'+ model + '/' + data_mode +'/dataset/'


train_imgs = load_data(drr_data_path + 'train_imgs.hdf5', 'train_imgs')
train_labs = load_data(drr_data_path + 'train_labs.hdf5', 'train_labs')



np.mean(np.abs(imgs[2]-imgs[3]))

X = np.moveaxis(train_imgs_c, 1, 3)
Z=Y=  np.moveaxis(train_labs, 1, 3)

Z=Y=np.zeros(X.shape, dtype='uint8')
#X = imgs[5][np.newaxis, ...]
Y = np.moveaxis(labs, 1, 3)
#Z = new_lab[np.newaxis, ...]
Z = Y

fig, ax = plt.subplots(1, 1)
extent = (0, X.shape[2], 0, X.shape[1])
tracker1 = track.IndexTracker(ax, X, Y, Z, extent, vmin = 0, vmax = 1)

fig.canvas.mpl_connect('key_press_event', tracker1.onpress)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress2)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress3)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress4)
fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)   
   
      
imgs = np.reshape(imgs,(4,181,768,1024))
imgs = np.log(imgs+1)
imgs = (imgs - np.min(imgs))/(np.max(imgs)-np.min(imgs))
imgs = imgs*2 - 1      
      
      
train_imgs = (train_imgs+1)/2.
train_imgs_c = np.power(train_imgs, 0.7)    
      
      
      
      
      
      
      
      
      