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
        q = random.uniform(0, 1)
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


def get_field(vxm_model, imgs_0, imgs_i, x_min, x_max, y_min, y_max, z_min, z_max):
    # get field
    val_input = [imgs_0[np.newaxis, x_min:x_max, y_min:y_max, z_min:z_max, np.newaxis], \
                 imgs_i[np.newaxis, x_min:x_max, y_min:y_max, z_min:z_max, np.newaxis]]
    val_pred = vxm_model.predict(val_input)
    field = sample([val_pred[1][0, ..., 0:3], val_pred[1][0, ..., 3:]])
    field =  np.stack([ct_resize(field[..., i], 2) for i in range(3)], axis = 3)[np.newaxis,...]

    field = np.pad(field, ((0, 0),
                           (x_min, imgs_0.shape[0]-x_max),
                           (y_min, imgs_0.shape[1]-y_max),
                           (z_min, imgs_0.shape[2]-z_max),
                           (0, 0)))
    
    return field



def vxm_phase_registration(transform_model, imgs_0, labs_0, imgs_i, labs_i, field, nr_deformations = 3):
    deformed_imgs, deformed_labs = [imgs_i], [labs_i]
    
    for i in range(nr_deformations):
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




def vxm_patient_registration(imgs, labs, body, vxm_model, patient_name, nr_deformations = 9, save_dir = 'D:/MasterAIThesis/h5py_data/online patient data/'):
    # only for the vumc patient
    imgs = imgs[..., :-16]
    labs = labs[..., :-16]
    body = body[..., :-16]  
    
    imgs[imgs>1000] = 1000
    imgs = (imgs/1000).astype('float32')
    
    
    x = np.where(body.any(axis = (1,2)))[0]
    y = np.where(body.any(axis = (0,2)))[0]
    z = np.where(body.any(axis = (0,1)))[0]
    
    x_min, x_max = x[0], x[-1] + 1
    y_min, y_max = y[0], y[-1] + 1
    z_min, z_max = z[0], z[-1] + 1
    
    x_min, x_max = crop16(x_min, x_max)
    y_min, y_max = crop16(y_min, y_max)
    z_min, z_max = crop16(z_min, z_max)
    
    transform_model = networks.nn_trf(imgs.shape[1:])

    
    for i in range(1, imgs.shape[0]):
        dirName = save_dir + patient_name + '/deformed_data/' + str(10*(i-1)) + '-' + str(10*i)
        if not os.path.exists(dirName):
            print("Create: " + dirName)    
            os.mkdir(dirName) 
            
        field = get_field(vxm_model, imgs[i-1], imgs[i], x_min, x_max, y_min, y_max, z_min, z_max)
        # get deformed imgs and labs
        deformed_imgs, deformed_labs = vxm_phase_registration(transform_model,
                                                              imgs[i-1], labs[i-1],
                                                              imgs[i], labs[i],                                                          
                                                              field, nr_deformations = nr_deformations)
        
        deformed_imgs = (deformed_imgs*1000).astype('int16')
        
        # save data
        dtn.save_data(dirName + '/', 'deformed_imgs', deformed_imgs)
        dtn.save_data(dirName + '/', 'deformed_labs', deformed_labs)
        print(i)
        
        


        
def pad16(a_min, a_max):
    l = a_max - a_min
    r = l%16
    if r>0:
        pad = 16 - r
        if pad%2 == 0:
            a_min = a_min - pad//2
            a_max = a_max + pad//2
        else:
            a_min = a_min - pad//2 - 1
            a_max = a_max + pad//2
            
    return a_min, a_max
        

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
        

# LOAD DATA        


load_data_path = 'D:/MasterAIThesis/h5py_data/vumc patient data/'   
patient_name = 'patient2'
imgs = dtn.load_data(load_data_path + patient_name + '/original_data/' + 'imgs.hdf5', 'imgs')
labs = dtn.load_data(load_data_path + patient_name + '/original_data/' + 'labs.hdf5', 'labs')

filename = load_data_path + patient_name + '/original_data/phase_0'
with h5py.File(filename, 'r') as f:
    body = np.array(f['BODY'], dtype = 'uint8')



imgs_r = imgs[..., :-16]
labs_r = labs[..., :-16]
body = body[..., :-16]


imgs_r = resize_all(imgs, 0.5)
labs_r = resize_all(labs, 0.5)
body_r = ct_resize(body, 0.8)


x = np.where(body.any(axis = (1,2)))[0]
y = np.where(body.any(axis = (0,2)))[0]
z = np.where(body.any(axis = (0,1)))[0]

x_min, x_max = x[0], x[-1] + 1
y_min, y_max = y[0], y[-1] + 1
z_min, z_max = z[0], z[-1] + 1

x_min, x_max = crop16(x_min, x_max)
y_min, y_max = crop16(y_min, y_max)
z_min, z_max = crop16(z_min, z_max)


imgs_r = imgs[:, x_min:x_max, y_min:y_max, z_min:z_max]
labs_r = labs[:, x_min:x_max, y_min:y_max, z_min:z_max]


imgs_r = imgs[:, 128:-128, 128:-128, 32:-32]
labs_r = labs[:, 128:-128, 128:-128, 32:-32]


imgs_r = imgs[:, 64:-64, 64:-64, 16:-16]
labs_r = labs[:, 64:-64, 64:-64, 16:-16]


imgs_r = imgs
labs_r = labs


imgs_r = resize_all(imgs, 0.75)
labs_r = resize_all(labs, 0.75)


imgs[imgs>1000] = 1000
imgs = (imgs/1000).astype('float32')

imgs_r[imgs_r>1000] = 1000
imgs_r = (imgs_r/1000).astype('float32')


# let's test it
train_generator = vxm_data_generator(imgs_r, 1)


# DEFINE MODEL
# our data will be of shape 160 x 192 x 224
vol_shape = [imgs_r.shape[1], imgs_r.shape[2], imgs_r.shape[3]]
ndims = 3

nf_enc = [16,32,32,32]
nf_dec = [32,32,32,32,16,16]


vxm_model = networks.miccai2018_net(vol_shape, nf_enc, nf_dec, 
                                    int_steps = 4,
                                    full_size = False, use_miccai_int = False)

image_sigma = 0.02
prior_lambda = 10
flow_vol_shape = vxm_model.outputs[-1].shape[1:-1]
loss_class = losses.Miccai2018(image_sigma, prior_lambda, flow_vol_shape=flow_vol_shape)

model_losses = [loss_class.recon_loss, loss_class.kl_loss]
loss_weights = [1, 1]



vxm_model = networks.cvpr2018_net(vol_shape, nf_enc, nf_dec, full_size = False)
model_losses = ['mse', losses.Grad('l2').loss]
# usually, we have to balance the two losses by a hyper-parameter.
lambda_param = 0.01
loss_weights = [10, lambda_param]




lr = 1e-4
decay_rate = lr / 10
momentum = 0.8
vxm_model.compile(optimizer=keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay_rate, nesterov=False),
                  loss=model_losses, 
                  loss_weights=loss_weights)




# load weights
vxm_model.load_weights('D:/MasterAIThesis/code/deform_data/lung_reg_final.h5')
vxm_model.load_weights('D:/MasterAIThesis/code/deform_data/voxelmorph/models/cvpr2018_vm2_cc.h5')


nb_epochs = 10
steps_per_epoch = 100
hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, 
                               steps_per_epoch=steps_per_epoch, verbose=1)


vxm_model.save_weights('deform_data/lung_reg_5.h5')

val_input = [imgs_r[0][np.newaxis,...,np.newaxis], imgs_r[5][np.newaxis,..., np.newaxis]]
val_pred = vxm_model.predict(val_input)
field = val_pred[1][0,...]
field =  np.stack([ct_resize(field[..., i], 2) for i in range(3)], axis = 3)[np.newaxis,...]

field = sample([val_pred[1][0, ..., 0:3], val_pred[1][0, ..., 3:]])
field =  np.stack([ct_resize(field[..., i], 2) for i in range(3)], axis = 3)[np.newaxis,...]


transform_model = networks.nn_trf(vol_shape)
pred_img_1 = val_pred[0][0,...,0]
pred_img = transform_model.predict([imgs_r[0][np.newaxis,...,np.newaxis], 
                                    field])[0,...,0]
pred_lab = transform_model.predict([labs_r[0][np.newaxis,...,np.newaxis].astype('float32'),
                                   field])[0,...,0]
pred_lab[pred_lab >= 0.5] = 1
pred_lab = pred_lab.astype('uint8')
new_lab = scipy.ndimage.binary_fill_holes(pred_lab)    
        
q = random.uniform(0, 1)
new_warp = q * (np.max(field)) + field   
  
q_prod = random.uniform(0, 1)
q_sum = random.uniform(-1, 1)
new_warp = q_prod * field + 20

pred_img_2 = transform_model.predict([imgs_r[0][np.newaxis,...,np.newaxis], 
                                    new_warp])[0,...,0] 

pred_lab_2 = transform_model.predict([labs_r[0][np.newaxis,...,np.newaxis].astype('float32'),
                                   new_warp])[0,...,0]
pred_lab_2[pred_lab_2 >= 0.5] = 1
pred_lab_2 = pred_lab_2.astype('uint8')
pred_lab_2 = scipy.ndimage.binary_fill_holes(pred_lab_2)



print(np.mean(np.abs(pred_img - pred_img_2))*1000)
print(np.mean(np.abs(pred_img_2 - imgs_r[1]))*1000)
print(np.mean(np.abs(pred_img_2 - imgs_r[0]))*1000)
print(np.mean(np.abs(imgs_r[1] - imgs_r[0]))*1000)




field_1 = bspline_registration(imgs[0].astype('float64'), imgs[5].astype('float64'),nr_samples = 100, voxel_spacing = 12, verbose = 1)

 
new_img = pirt.deform_backward(imgs[0].astype('float64'), field_1, order = 1).astype('int16') 
new_lab = pirt.deform_backward(labs[0].astype('float64'), field_1, order = 1)
new_lab[new_lab >= 0.5] = 1
new_lab = new_lab.astype('uint8')


np.mean(np.abs(imgs[1] - imgs[2]))
    



load_data_path = 'D:/MasterAIThesis/h5py_data/vumc patient data/'   
patient_name = 'patient1'
i = 5
load_dir = load_data_path + patient_name + '/deformed_data/' + str(10*(i-1)) + '-' + str(10*i) + '/'

imgs = dtn.load_data(load_dir + 'deformed_imgs.hdf5', 'deformed_imgs')
labs = dtn.load_data(load_dir + 'deformed_labs.hdf5', 'deformed_labs')



from skimage import morphology
aux = labs.copy()
for i in range(labs.shape[0]):
    aux[i] =  morphology.remove_small_objects(aux[i].astype('bool'), connectivity = 3)

# plot slices

X = imgs
#X = pred_img_2[np.newaxis, ...]*1000
#Y = pred_lab[np.newaxis, ...]
Y = labs
#Z = pred_lab_2[np.newaxis, ...]
Z = labs

fig, ax = plt.subplots(1, 1)
extent = (0, imgs.shape[2], 0, imgs.shape[1])
tracker1 = track.IndexTracker(ax, X, Y, Z, extent, vmin = -1000, vmax = 1000)

fig.canvas.mpl_connect('key_press_event', tracker1.onpress)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress2)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress3)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress4)
fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)





