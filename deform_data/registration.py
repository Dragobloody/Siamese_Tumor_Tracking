import sys
# local imports.
sys.path.append('D:/MasterAIThesis/code/deform_data/voxelmorph/ext/pynd-lib/')
sys.path.append('D:/MasterAIThesis/code/deform_data/voxelmorph/ext/pytools-lib/')
sys.path.append('D:/MasterAIThesis/code/deform_data/voxelmorph/ext/neuron/')
sys.path.append('D:/MasterAIThesis/code/deform_data/voxelmorph/src/')
sys.path.append('D:/MasterAIThesis/code/deform_data/voxelmorph/')
sys.path.append('D:/MasterAIThesis/code/deform_data/')
sys.path.append('D:/MasterAIThesis/code/load_dicom/')
import numpy as np
import dicom_to_numpy as dtn
import os
import pirt
import random
import index_tracker as track   
import matplotlib.pyplot as plt
import keras.layers
import scipy
import h5py


from src import networks, losses
import neuron


def ndgrid(*args, **kwargs):
    """
    Disclaimer: This code is taken directly from the scitools package [1]
    Since at the time of writing scitools predominantly requires python 2.7 while we work with 3.5+
    To avoid issues, we copy the quick code here.
    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args, **kwargs)


def bw_grid(vol_shape, spacing, thickness=1):
    """
    draw a black and white ND grid.
    Parameters
    ----------
        vol_shape: expected volume size
        spacing: scalar or list the same size as vol_shape
    Returns
    -------
        grid_vol: a volume the size of vol_shape with white lines on black background
    """

    # check inputs
    if not isinstance(spacing, (list, tuple)):
        spacing = [spacing] * len(vol_shape)
    assert len(vol_shape) == len(spacing)

    # go through axes
    grid_image = np.zeros(vol_shape)
    for d, v in enumerate(vol_shape):
        rng = [np.arange(0, f) for f in vol_shape]
        for t in range(thickness):
            rng[d] = np.append(np.arange(0+t, v, spacing[d]), -1)
            grid_image[ndgrid(*rng)] = 1

    return grid_image


def vxm_data_generator(x_data, batch_size=32):
    """
    generator that takes in data of size [N, H, W], and yields data for our vxm model
    
    Note that we need to provide numpy data for each input, and each output
    
    inputs:  moving_image [bs, H, W, 1], fixed_image [bs, H, W, 1]
    outputs: moved_image  [bs, H, W, 1], zeros [bs, H, W, 2]
    """
    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation. We'll explain this below.
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs
        # inputs need to be of the size [batch_size, H, W, number_features]
        #   number_features at input is 1 for us
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # outputs
        # we need to prepare the "true" moved image.  
        # Of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]
        
        yield inputs, outputs 


def sample(args):
    """
    sample from a normal distribution
    """
    mu = args[0]
    log_sigma = args[1]
    noise = np.random.normal(0.0, 1.0, mu.shape).astype('float32')
    z = mu + np.exp(log_sigma/2.0) * noise
    return z



def ct_resize(data, zoom, cval=0.0):
	"""
		dataset - Input ct scans, must be a 4d numpy array 
		zoom - Zoom factor , float or array of floats for each dimensions
		cval - Value used for points outside the boundaries of the input
	"""

	out_data = scipy.ndimage.interpolation.zoom(data, zoom = zoom, cval=cval)

	return out_data

def resize_all(dataset, zoom, cval = 0.0):
	"""
		dataset - Input ct scans, must be a 4d numpy array 
		zoom - Zoom factor , float or array of floats for each dimensions
		cval - Value used for points outside the boundaries of the input
	"""
	return np.stack([ct_resize(dataset[i], zoom, cval) for i in range(dataset.shape[0])], axis = 0)



def pad_data(data, padding, mode = 'edge'):
  return np.pad(data, padding, mode = mode)


def crop16(a_min, a_max):
    l = a_max - a_min
    crop = l%16
    if crop>0:        
        if crop%2 == 0:
            a_min = a_min + crop//2
            a_max = a_max - crop//2
        else:
            a_min = a_min + crop//2 + 1
            a_max = a_max - crop//2
            
    return a_min, a_max


def bspline_registration(moving, fixed, nr_samples = 10000, iterations = 2000, voxel_spacing = 8, verbose = 0):
  # register
  bspline_reg = pirt.ElastixRegistration_affine(moving, fixed)
  bspline_params = bspline_reg.params
  bspline_params.NumberOfSpatialSamples = nr_samples
  bspline_params.MaximumNumberOfIterations = iterations
  bspline_params.FinalGridSpacingInVoxels = voxel_spacing
  bspline_reg.set_params(bspline_params)
  bspline_reg.register(verbose = verbose)
  # get field
  bspline_field = bspline_reg.get_final_deform(mapping = 'backward')
  field = np.array([bspline_field[2], bspline_field[1], bspline_field[0]], dtype = 'float64')

  return field


def phase_registration(imgs_0, labs_0, imgs_i, labs_i, field, nr_deformations = 3):
    
    deformed_imgs, deformed_labs = [imgs_i], [labs_i]
    for i in range(nr_deformations):
        q = random.uniform(0.2, 1)
        new_warp = q * field
         
        new_img = pirt.deform_backward(imgs_0.astype('float64'), new_warp, order = 1).astype('int16') 
        new_lab = pirt.deform_backward(labs_0.astype('float64'), new_warp, order = 1)
        new_lab[new_lab >= 0.5] = 1
        new_lab = new_lab.astype('uint8')
        
        deformed_imgs.append(new_img)
        deformed_labs.append(new_lab)        
    
    deformed_imgs = np.stack(deformed_imgs, axis= 0).astype('int16') 
    deformed_labs = np.stack(deformed_labs, axis= 0).astype('uint8') 
    
    return deformed_imgs, deformed_labs




def sample_field(val_pred):
    # get field    
    field = sample([val_pred[1][0, ..., 0:3], val_pred[1][0, ..., 3:]])
    field =  np.stack([ct_resize(field[..., i], 2) for i in range(3)], axis = 3)[np.newaxis,...]
    
    return field



def vxm_phase_registration(transform_model, imgs_0, labs_0, imgs_i, labs_i, 
                           x_min, x_max, y_min, y_max, z_min, z_max,
                           val_pred, nr_deformations = 3):
    deformed_imgs, deformed_labs = [imgs_0], [labs_0]
    
    for i in range(nr_deformations):
        # sample fields
        field = sample_field(val_pred)
        field = np.pad(field, ((0, 0),
                               (x_min, imgs_0.shape[0]-x_max),
                               (y_min, imgs_0.shape[1]-y_max),
                               (z_min, imgs_0.shape[2]-z_max),
                               (0, 0)))       
        
        q = random.uniform(0, 1)
        new_warp = q * field 
         
        new_img = transform_model.predict([imgs_0[np.newaxis,...,np.newaxis], 
                                    new_warp])[0,...,0]
        new_lab = transform_model.predict([labs_0[np.newaxis,...,np.newaxis].astype('float32'),
                                           new_warp])[0,...,0]
            
        new_lab[new_lab >= 0.5] = 1
        new_lab = new_lab.astype('uint8')
        new_lab = scipy.ndimage.binary_fill_holes(new_lab)
        
        deformed_imgs.append(new_img)
        deformed_labs.append(new_lab)        
    
    deformed_imgs = np.stack(deformed_imgs, axis= 0).astype('float32') 
    deformed_labs = np.stack(deformed_labs, axis= 0).astype('uint8') 
    
    return deformed_imgs, deformed_labs




def get_body_idx(body):
    x = np.where(body.any(axis = (1,2)))[0]
    y = np.where(body.any(axis = (0,2)))[0]
    z = np.where(body.any(axis = (0,1)))[0]    
    x_min, x_max = x[0], x[-1] + 1
    y_min, y_max = y[0], y[-1] + 1
    z_min, z_max = z[0], z[-1] + 1    
    x_min, x_max = crop16(x_min, x_max)
    y_min, y_max = crop16(y_min, y_max)
    z_min, z_max = crop16(z_min, z_max)
    
    return  x_min, x_max, y_min, y_max, z_min, z_max



def define_vxm_model(x_min, x_max, y_min, y_max, z_min, z_max):
    # DEFINE MODEL    
    vol_shape = [x_max - x_min, y_max - y_min, z_max - z_min]
    
    nf_enc = [16,32,32,32]
    nf_dec = [32,32,32,32,16,16]
    
    vxm_model = networks.miccai2018_net(vol_shape, nf_enc, nf_dec, 
                                        int_steps = 4,
                                        full_size = False, use_miccai_int = False)
    # load weights
    vxm_model.load_weights('D:/MasterAIThesis/code/deform_data/lung_reg_final.h5')

    return vxm_model


def vxm_patient_registration(imgs, labs, body, patient_name, nr_deformations = 9, save_dir = 'D:/MasterAIThesis/h5py_data/online patient data/'):
    # only for the vumc patient
    imgs = imgs
    labs = labs
    body = body[..., :-16]  
    
    # get data between 0 and 1
    imgs[imgs>1000] = 1000
    imgs = (imgs/1000).astype('float32')
    
    # get body boundaries
    x_min, x_max, y_min, y_max, z_min, z_max = get_body_idx(body)

    # load voxelmorph model
    vxm_model = define_vxm_model(x_min, x_max, y_min, y_max, z_min, z_max)   
    
    # define transform model
    transform_model = networks.nn_trf(imgs.shape[1:]) 
    
    # register phases
    for i in range(0, imgs.shape[0]):
        dirName = save_dir + patient_name + '/deformed_data/' + str(10*i) + '-' + str(10*(i+1))
        if not os.path.exists(dirName):
            print("Create: " + dirName)    
            os.mkdir(dirName)             
        
        j = (i+4)%10
        val_input = [imgs[i][np.newaxis, x_min:x_max, y_min:y_max, z_min:z_max, np.newaxis], \
                 imgs[j][np.newaxis, x_min:x_max, y_min:y_max, z_min:z_max, np.newaxis]]
        val_pred = vxm_model.predict(val_input)            
            
        # get deformed imgs and labs
        deformed_imgs, deformed_labs = vxm_phase_registration(transform_model,
                                                              imgs[i], labs[i],
                                                              imgs[j], labs[j], 
                                                              x_min, x_max, 
                                                              y_min, y_max, 
                                                              z_min, z_max,
                                                              val_pred,
                                                              nr_deformations = nr_deformations)
        
        deformed_imgs = (deformed_imgs*1000).astype('int16')
        
        # save data
        dtn.save_data(dirName + '/', 'deformed_imgs', deformed_imgs)
        dtn.save_data(dirName + '/', 'deformed_labs', deformed_labs)
        print(i)
        
        
"""
# LOAD DATA       
load_data_path = 'D:/MasterAIThesis/h5py_data/vumc patient data/'   
patient_name = 'patient1'
imgs = dtn.load_data(load_data_path + patient_name + '/original_data/' + 'imgs.hdf5', 'imgs')
labs = dtn.load_data(load_data_path + patient_name + '/original_data/' + 'labs.hdf5', 'labs')

filename = load_data_path + patient_name + '/original_data/phase_0'
with h5py.File(filename, 'r') as f:
    body = np.array(f['body'], dtype = 'uint8')



vxm_patient_registration(imgs, labs, body, 
                         patient_name, 
                         nr_deformations = 3, 
                         save_dir = load_data_path)





load_data_path = 'D:/MasterAIThesis/h5py_data/vumc patient data/'   
patient_name = 'patient1'
imgs = dtn.load_data(load_data_path + patient_name + '/deformed_data/90-100/' + 'deformed_imgs.hdf5', 'deformed_imgs')
labs = dtn.load_data(load_data_path + patient_name + '/deformed_data/90-100/' + 'deformed_labs.hdf5', 'deformed_labs')

X = np.flip(np.transpose(image, (2, 1, 0)), axis=0)[np.newaxis, ...]
Y = Z = np.zeros(X.shape, dtype = 'uint8')

X = np.transpose(imgs, (0, 3,2,1))
Z=Y = np.transpose(labs, (0, 3,2,1))

X = np.flip(np.transpose(imgs_0, (2, 1, 0)), axis=0)[np.newaxis, ...]
Z=Y= np.flip(np.transpose(labs_0, (2, 1, 0)), axis=0)[np.newaxis, ...]

X = np.flip(np.transpose(imgs_i, (2, 1, 0)), axis=0)[np.newaxis, ...]
Z=Y= np.flip(np.transpose(labs_i, (2, 1, 0)), axis=0)[np.newaxis, ...]

X = np.flip(np.transpose(new_img, (2, 1, 0)), axis=0)[np.newaxis, ...]
Z=Y= np.flip(np.transpose(new_lab, (2, 1, 0)), axis=0)[np.newaxis, ...]

X = np.flip(np.transpose(deformed_grid, (2, 1, 0)), axis=0)[np.newaxis, ...]
Y = Z = np.zeros(X.shape, dtype = 'uint8')

X = np.transpose(field[0], (3, 2, 1, 0))
Y = Z = np.zeros(X.shape, dtype = 'uint8')

X = np.flip(np.transpose(plot_imgs, (2, 1, 0)), axis=0)[np.newaxis, ...]
Y = Z = np.flip(np.transpose(plot_labs, (2, 1, 0)), axis=0)[np.newaxis, ...]

fig, ax = plt.subplots(1, 1)
extent = (0, X.shape[2], 0, X.shape[2])
tracker1 = track.IndexTracker(ax, X, Y, Z, extent, vmin=0, vmax=1)

fig.canvas.mpl_connect('key_press_event', tracker1.onpress)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress2)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress3)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress4)
fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)
   


plot_imgs = np.array([imgs_0[291], imgs_i[291], new_img[291], deformed_grid[291]])
plot_labs = np.array([labs_0[291], labs_i[291], new_lab[291], np.zeros(deformed_grid[291].shape, dtype='uint8')])


grid = bw_grid((512, 512, 144), 8)
deformed_grid = transform_model.predict([grid[np.newaxis,...,np.newaxis].astype('uint8'),
                                           new_warp])[0,...,0]
deformed_grid[deformed_grid >= 0.5] = 1
deformed_grid = deformed_grid.astype('uint8')

"""