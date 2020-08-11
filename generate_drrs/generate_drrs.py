import os, sys
sys.path.append('D:/MasterAIThesis/code/load_dicom')
sys.path.append('D:/MasterAIThesis/code/generate_drrs')
sys.path.append('D:/MasterAIThesis/code/generate_drrs/deepdrr')

import numpy as np

import matplotlib.pyplot as plt
import h5py
import copy
import random


import dicom_to_numpy as dtn
from deepdrr import projector
import projection_matrix
from analytic_generators import add_noise
import mass_attenuation_gpu as mass_attenuation
import spectrum_generator
import add_scatter
from utils import Camera
import segmentation
import get_tumor_center as gtc

import index_tracker as track   



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


def compute_tumor_drr_ratio(mask):
    x = np.where(mask.any(axis = 1))[0]
    y = np.where(mask.any(axis = 0))[0]
    height = x.shape[0]
    width = y.shape[0]
    
    tumor_area = height*width
    drr_area = mask.shape[0]*mask.shape[1]
    ratio = np.sqrt(tumor_area/drr_area)
    
    return ratio


def compute_final_ratio(mask, final_tumor_size = 100):
    tumor_area = final_tumor_size*final_tumor_size
    drr_area = mask.shape[0]*mask.shape[1]
    
    return np.sqrt(tumor_area/drr_area)

def conv_hu_to_density(hu_values, smoothAir = False):
    #Use two linear interpolations from data: (HU,g/cm^3)
    # use for lower HU: density = 0.001029*HU + 1.03
    # use for upper HU: density = 0.0005886*HU + 1.03

    #set air densities
    if smoothAir:
        hu_values[hu_values <= -900] = -1000;    
    densities = np.maximum(np.minimum(0.001029 * hu_values + 1.030, 0.0005886 * hu_values + 1.03), 0);
    return densities.astype(np.float32)

def conv_hu_to_materials_thresholding(hu_values):
    print("segmenting volume with thresholding")
    materials = {}
    # Air
    materials["air"] = hu_values <= -800
    # Soft Tissue
    materials["soft tissue"] = (-800 < hu_values) * (hu_values <= 350)
    # Bone
    materials["bone"] = (350 < hu_values)
    return materials

def conv_hu_to_materials(hu_values):
    print("segmenting volume with Vnet")
    segmentation_network = segmentation.SegmentationNet()
    materials = segmentation_network.segment(hu_values, show_results = False)
    return materials
  

def get_materials(volume, use_thresholding_segmentation = True):
  #convert hu_values to materials
  if not use_thresholding_segmentation:
      materials = conv_hu_to_materials(volume)
  else:
      materials = conv_hu_to_materials_thresholding(volume)
  return materials


def compute_densities_and_materials(data, use_thresholding_segmentation = True):
  densities = []
  materials = []
  for i in range(data.shape[0]):
    density = conv_hu_to_density(data[i])
    material = get_materials(data[i], use_thresholding_segmentation)
    densities.append(density)
    materials.append(material)
  densities = np.stack(densities, axis = 0)
  
  return densities.astype('float32'), materials


def shifting(img, x_shift, y_shift):  
  min_val = np.min(img)
  shifted_img = np.roll(img, (x_shift, y_shift), axis=(0, 1))

  if x_shift >= 0:
    shifted_img[:x_shift, :] = min_val
  else:
    shifted_img[x_shift:, :] = min_val    

  if y_shift >= 0:
    shifted_img[:, :y_shift] = min_val
  else:
    shifted_img[:, y_shift:] = min_val  

  return shifted_img


def crop_around_center(images, new_height = 768, new_width = 1024):
    width = images.shape[2]
    height = images.shape[1]
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    
    return images[:, top:bottom, left:right]



def shift_drrs(forward_projections, target_projection, nr_shifts = 4, new_height = 768, new_width = 1024, decorrelation = True):
    # shift soft tissue for bone decorrelation
    images = {}
    images['air'] = [forward_projections['air']]
    images['bone'] = [forward_projections['bone']]
    images['soft tissue'] = [forward_projections['soft tissue']]
    targets = [target_projection]
    
    for i in range(nr_shifts):
        shifted_forward_projections = copy.deepcopy(forward_projections)
        shifted_target_projection = copy.deepcopy(target_projection)
        
        for j in range(target_projection.shape[0]):
            x_shift = random.randint(-max_x, max_x)
            y_shift = random.randint(-max_y, max_y)
            
            # if decorrelation = False then do only shifting
            if not decorrelation:
                shifted_forward_projections['bone'][j] = shifting(shifted_forward_projections['bone'][j], x_shift, y_shift)
                
            shifted_forward_projections['air'][j] = shifting(shifted_forward_projections['air'][j], x_shift, y_shift)
            shifted_forward_projections['soft tissue'][j] = shifting(shifted_forward_projections['soft tissue'][j], x_shift, y_shift)
            shifted_target_projection[j] = shifting(shifted_target_projection[j], x_shift, y_shift)
            
        images['air'].append(shifted_forward_projections['air'])
        images['bone'].append(shifted_forward_projections['bone'])
        images['soft tissue'].append(shifted_forward_projections['soft tissue'])
        targets.append(shifted_target_projection)
    
    images['air'] = np.concatenate(images['air'], axis = 0)
    images['bone'] = np.concatenate(images['bone'], axis = 0)
    images['soft tissue'] = np.concatenate(images['soft tissue'], axis = 0)
    targets = np.concatenate(targets, axis = 0)
    
    # get the center 
    images['air'] = crop_around_center(images['air'], new_height, new_width)
    images['bone'] = crop_around_center(images['bone'], new_height, new_width)
    images['soft tissue'] = crop_around_center(images['soft tissue'], new_height, new_width)
    targets = crop_around_center(targets, new_height, new_width)       
    
    return images, targets


def generate_projections_on_sphere(volume, materials, target_mask, voxel_size, 
                                   min_theta, max_theta, min_phi, max_phi,
                                   spacing_theta, spacing_phi, photon_count, 
                                   camera, spectrum, scatter = False, noise = True, 
                                   standard = True, decorrelate = True,
                                   max_x = 38, max_y = 50, nr_shifts = 4, origin = [0,0,0]):
        
    new_height = camera.sensor_height - 2*max_x
    new_width = camera.sensor_width - 2*max_y    
    # generate angle pairs on a sphere
    thetas, phis = projection_matrix.generate_uniform_angels(min_theta, max_theta, min_phi, max_phi, spacing_theta, spacing_phi)
    # generate projection matrices from angles
    proj_mats = projection_matrix.generate_projection_matrices_from_values(camera.source_to_detector_distance, camera.pixel_size, camera.pixel_size, camera.sensor_width, camera.sensor_height, camera.isocenter_distance, phis, thetas)
    
        
    #forward project densities
    forward_projections = projector.generate_projections(proj_mats, volume, materials, origin, voxel_size, camera.sensor_width, camera.sensor_height, mode="linear", max_blockind=200, threads=8)
    target_projection = projector.generate_target_projection(proj_mats, volume, (target_mask > 0), origin, voxel_size, camera.sensor_width, camera.sensor_height, mode="linear", max_blockind=200, threads=8)
    
    #forward_projections['bone'] = forward_projections['bone']*2
    
    # shift soft tissue for bone decorrelation
    images_standard, targets_standard, images_decorrelated, targets_decorrelated = None, None, None, None
    
    #add scatter
    if scatter:
        scatter_net = add_scatter.ScatterNet()                
        
    # do not shift or decorrelate
    if standard:
        images_standard = copy.deepcopy(forward_projections)
        targets_standard = copy.deepcopy(target_projection)
        
        images_standard['air'] = crop_around_center(images_standard['air'], new_height, new_width)
        images_standard['bone'] = crop_around_center(images_standard['bone'], new_height, new_width)
        images_standard['soft tissue'] = crop_around_center(images_standard['soft tissue'], new_height, new_width)
        targets_standard = crop_around_center(targets_standard, new_height, new_width)
        # calculate intensity at detector (images: mean energy one photon emitted from the source deposits at the detector element, photon_prob: probability of a photon emitted from the source to arrive at the detector)
        images_standard, photon_prob = mass_attenuation.calculate_intensity_from_spectrum(images_standard, spectrum)
        
        if scatter:
            scatter_standard = scatter_net.add_scatter(images_standard, camera)
            photon_prob *= 1 + scatter_standard/images_standard
            images_standard += scatter_standard
        
        #add poisson noise
        if noise:
            images_standard = add_noise(images_standard, photon_prob, photon_count)
        
    
    # shift and decorrelate    
    if decorrelate:
        images_decorrelated, targets_decorrelated = shift_drrs(forward_projections, target_projection, nr_shifts, new_height, new_width, decorrelation = True)
        # calculate intensity at detector (images: mean energy one photon emitted from the source deposits at the detector element, photon_prob: probability of a photon emitted from the source to arrive at the detector)
        images_decorrelated, photon_prob_decorrelated = mass_attenuation.calculate_intensity_from_spectrum(images_decorrelated, spectrum)
        #add scatter
        if scatter:           
            scatter_decorrelated = scatter_net.add_scatter(images_decorrelated, camera)        
            photon_prob_decorrelated *= 1 + scatter_decorrelated/images_decorrelated
            images_decorrelated += scatter_decorrelated
        #add poisson noise
        if noise:
            images_decorrelated = add_noise(images_decorrelated, photon_prob_decorrelated, photon_count)   
    
    return images_standard, targets_standard, images_decorrelated, targets_decorrelated





def generate_drrs(img, materials, lab, voxel_size, min_theta, max_theta, min_phi, 
                  max_phi, spacing_theta, spacing_phi, photon_count, camera, spectrum, 
                  scatter = False, noise = True,
                  standard = True, decorrelate = True,
                  max_x = 38, max_y = 50, nr_shifts = 4, origin = [0,0,0]):
    
  drrs_standard, masks_standard, \
  drrs_decorrelated, masks_decorrelated = generate_projections_on_sphere(img, materials, lab, voxel_size, min_theta, max_theta, min_phi, 
                                                  max_phi, spacing_theta, spacing_phi, photon_count, camera, spectrum, 
                                                  scatter = scatter, noise = noise,
                                                  standard = standard, decorrelate = decorrelate,
                                                  max_x = max_x, max_y = max_y, nr_shifts = nr_shifts, origin = origin)
  if standard:
      masks_standard[masks_standard>0] = 1
 
  if decorrelate:
      masks_decorrelated[masks_decorrelated>0] = 1
  
  
  return drrs_standard, masks_standard, drrs_decorrelated, masks_decorrelated


def generate_patient_drrs(data_path, patient_name, voxel_size, min_theta, max_theta, min_phi, 
                          max_phi, spacing_theta, spacing_phi, photon_count, camera, spectrum, 
                          scatter = False, noise = True,
                          standard = True, decorrelate = True,
                          max_x = 38, max_y = 50, nr_shifts = 4, origin = [0,0,0]):
  
  k = os.listdir(data_path + patient_name + '/deformed_data/')
  for i in range(len(k)): 
    load_dir = data_path + patient_name + '/deformed_data/' + str(10*i) + '-' + str(10*(i+1)) + '/'
    standard_save_dir = data_path + patient_name + '/DRRs/standard/' + str(10*i) + '-' + str(10*(i+1))
    decorrelated_save_dir = data_path + patient_name + '/DRRs/decorrelated/' + str(10*i) + '-' + str(10*(i+1)) 

    if not os.path.exists(standard_save_dir):
      os.mkdir(standard_save_dir)  
    if not os.path.exists(decorrelated_save_dir):
      os.mkdir(decorrelated_save_dir)
    standard_save_dir = standard_save_dir + '/'
    decorrelated_save_dir = decorrelated_save_dir + '/'
    
    # load data
    imgs = load_data(load_dir + 'deformed_imgs.hdf5','deformed_imgs')
    labs = load_data(load_dir + 'deformed_labs.hdf5','deformed_labs')
    imgs = np.flip(imgs, axis = 3)    
    labs = np.flip(labs, axis = 3) 
    
     
    standard_drr_imgs, standard_drr_labs = [], []
    decorrelated_drr_imgs, decorrelated_drr_labs = [], []

    # get densities and materials
    imgs, materials = compute_densities_and_materials(imgs.astype('float32'),
                                                      use_thresholding_segmentation = False)

    # generate drrs
    for j in range(imgs.shape[0]):
      drrs_standard, masks_standard, \
      drrs_decorrelated, masks_decorrelated = generate_drrs(imgs[j], materials[j], labs[j], voxel_size, min_theta, max_theta, min_phi, 
                                                             max_phi, spacing_theta, spacing_phi, photon_count, camera, spectrum, 
                                                             scatter = False, noise = True,
                                                             standard = standard, decorrelate = decorrelate,
                                                             max_x = max_x, max_y = max_y, nr_shifts = nr_shifts, origin = origin)
  
      if standard:
          standard_drr_imgs.append(drrs_standard)
          standard_drr_labs.append(masks_standard)
     
      if decorrelate:
          decorrelated_drr_imgs.append(drrs_decorrelated)
          decorrelated_drr_labs.append(masks_decorrelated)
          
    if standard:
        standard_drr_imgs = np.concatenate(standard_drr_imgs, axis= 0).astype('float32')   
        standard_drr_labs = np.concatenate(standard_drr_labs, axis= 0).astype('uint8')    
        # save data
        save_data(standard_save_dir, 'standard_drr_imgs', standard_drr_imgs)
        save_data(standard_save_dir, 'standard_drr_labs', standard_drr_labs)        
   
        
    if decorrelate:
        decorrelated_drr_imgs = np.concatenate(decorrelated_drr_imgs, axis= 0).astype('float32')   
        decorrelated_drr_labs = np.concatenate(decorrelated_drr_labs, axis= 0).astype('uint8')    
        # save data
        save_data(decorrelated_save_dir, 'decorrelated_drr_imgs', decorrelated_drr_imgs)
        save_data(decorrelated_save_dir, 'decorrelated_drr_labs', decorrelated_drr_labs)    
    
    print(i)
    

    
"""
# SET HYPERPARAMETERS   
#2x2 binning
max_x = 32
max_y = 32
nr_shifts = 9
standard = True
decorrelate = True
camera = Camera(sensor_width = int(1024) + 2*max_y, sensor_height = int(768) + 2*max_x, 
                pixel_size = 0.388, source_to_detector_distance = 1500, 
                isocenter_distance = 1000)

# Projection parameter
min_theta = 270
max_theta = 271
min_phi = 90
max_phi = 271
spacing_theta = 30
spacing_phi = 1
photon_count = 500000
#origin [0,0,0] corresponds to the center of the volume
spectrum = spectrum_generator.SPECTRUM120KV_AL43
  


data_path = 'D:/MasterAIThesis/h5py_data/vumc phantom data/'
for i in range(1, 2):
    patient_name = 'phantom' + str(i)
    # get voxel size
    filename = data_path + patient_name + '/original_data/phase_0'
    with h5py.File(filename, 'r') as f:
        voxel_size = np.array(f['spacing'], dtype = 'float32')
        
    #voxel_size = voxel_size/voxel_size[2]
    
    # get tumor center
    labs = dtn.load_data(data_path + patient_name + '/original_data/' + 'labs.hdf5', 'labs')    
    labs = np.flip(labs, axis = 3)    
    #imgs = dtn.load_data(data_path + patient_name + '/original_data/' + 'imgs.hdf5', 'imgs')    
    #imgs = np.flip(imgs, axis = 3)
    
    x, y, z = gtc.get_patient_relative_center(labs)
    origin = [x, y, z] * voxel_size
    # generate drrs
    generate_patient_drrs(data_path, patient_name, voxel_size, min_theta, max_theta, min_phi, 
                              max_phi, spacing_theta, spacing_phi, photon_count, camera, spectrum, 
                              scatter = False, noise = True,
                              standard = standard, decorrelate = decorrelate,
                              max_x = max_x, max_y = max_y, nr_shifts = nr_shifts, origin = origin)
    




import scipy

load_dir = 'D:/MasterAIThesis/h5py_data/vumc phantom data/phantom2_2/DRRs/decorrelated/0-10/'
imgs = load_data(load_dir + 'decorrelated_drr_imgs.hdf5','decorrelated_drr_imgs')
labs = load_data(load_dir + 'decorrelated_drr_labs.hdf5','decorrelated_drr_labs')




drrs = np.log(drrs_standard+np.min(drrs_standard))
drrs = (drrs - np.min(drrs))/(np.max(drrs)-np.min(drrs))
drrs_standard = drrs_standard*2 - 1




plt.imshow(masks_standard[2, ...], cmap = 'gray')

plt.imshow(drrs_standard[2, 384//2:384+384//2, 512//2:512+512//2], cmap = 'gray')
plt.figure()
plt.imshow(masks_standard[2, 384//2:384+384//2, 512//2:512+512//2], cmap = 'gray')
plt.figure()
plt.imshow(masks[12], cmap = 'gray')



load_dir = 'D:/MasterAIThesis/h5py_data/vumc patient data/patient2_2/DRRs/standard/0-10/'
imgs = load_data(load_dir + 'standard_drr_imgs.hdf5','standard_drr_imgs')
labs = load_data(load_dir + 'standard_drr_labs.hdf5','standard_drr_labs')



imgs = np.moveaxis(imgs, 0, 2)[np.newaxis, ...]
labs = np.moveaxis(labs, 0, 2)[np.newaxis, ...]

X = imgs
Z = Y = labs

X = imgs[:, 384//2:384+384//2, 512//2:512+512//2, :]
Y = labs[:, 384//2:384+384//2, 512//2:512+512//2, :]
Z = labs[:, 384//2:384+384//2, 512//2:512+512//2, :]

X = np.transpose(imgs,(0, 3, 2, 1))
Z = Y = np.transpose(labs,(0, 3, 2, 1))

X = np.transpose(xray_imgs, (3, 1, 2, 0))
Z = Y = np.zeros(X.shape, dtype = 'uint8')

X = np.transpose(drrs, (1, 2, 0))[np.newaxis, ...]
Z=Y = np.transpose(masks_standard, (1, 2, 0)).astype('uint8')[np.newaxis, ...]
#Z = Y = np.zeros(X.shape, dtype='uint8')

X = np.transpose(aux1, (1, 2, 0))[np.newaxis, ...]
Z=Y = np.transpose(masks_standard, (1, 2, 0)).astype('uint8')[np.newaxis, ...]

fig, ax = plt.subplots(1, 1)
extent = (0, X.shape[2], 0, X.shape[1])
tracker1 = track.IndexTracker(ax, X, Y, Z, extent, vmin=0, vmax=2)

fig.canvas.mpl_connect('key_press_event', tracker1.onpress)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress2)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress3)
fig.canvas.mpl_connect('key_press_event', tracker1.onpress4)
fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)
   


xrays = imgs
# data normalization
mean = np.mean(xrays)
std = np.std(xrays)
xrays = (xrays - mean) / std  
# data normalization
mean = np.mean(aux2)
std = np.std(aux2)
aux2 = (aux2 - mean) / std  

plt.imshow(xrays[106], cmap = 'gray', vmax = 0.2)
plt.figure()
plt.imshow(drrs_standard[90], cmap = 'gray', vmax = 0)


#-------------------------------------------------------------------

drrs = drrs_standard[:, 192:-192, 256:-256]
masks = masks_standard[:, 192:-192, 256:-256]
c = 1/np.log(1 + np.max(drrs_standard, axis = (1,2)))
drrs = np.log(drrs_standard+1)
drrs = np.multiply(c[..., np.newaxis, np.newaxis], drrs)

c = 1/np.log(1 + np.max(drrs_decorrelated, axis = (1,2)))
drrs_decorrelated = np.log(drrs_decorrelated+1)
drrs_decorrelated = np.multiply(c[..., np.newaxis, np.newaxis], drrs_decorrelated)




aux1 = np.power(drrs, 0.7)
plt.imshow(aux2[9], cmap='gray')
plt.figure()
plt.imshow(masks_standard[9], cmap='gray')



c = 1/np.log(1 + np.max(xrays/1000, axis = (1,2)))
aux3 = np.log(xrays/1000+1)
aux3 = np.multiply(c[..., np.newaxis, np.newaxis], aux3)
aux3 = aux3*2 - 1

aux3 = (xrays - np.min(xrays))/(np.max(xrays)-np.min(xrays))
aux4 = np.power(aux3, 0.7)

aux5 = (test_imgs - np.min(test_imgs))/(np.max(test_imgs)-np.min(test_imgs))
aux6 = np.power(aux5, 0.1)



aux1_tf = tf.convert_to_tensor(drrs_standard[90])
aux1_tf = tf.image.adjust_gamma(aux1_tf, 0.4)
aux1_0 = aux1_tf.eval()


aux1 = np.log(drrs_standard[:333]+np.mean(drrs_standard[:333]))
aux1 = (aux1-np.min(aux1))/(np.max(aux1)-np.min(aux1))
aux1 = aux1*2 - 1



aux2 = np.log(drrs_standard+np.min(drrs_standard)+1)
aux5 = (drrs_standard - np.min(drrs_standard))/(np.max(drrs_standard)-np.min(drrs_standard))
aux5 = aux5*2 - 1


aux3 = np.log(xrays+np.mean(xrays))
aux3 = (aux3-np.min(aux3))/(np.max(aux3)-np.min(aux3))
aux3 = aux3*2 - 1

aux4 = (xrays - np.min(xrays))/(np.max(xrays)-np.min(xrays))
aux4 = aux4*2 - 1


y = (np.mean(xrays)-np.mean(drrs_standard))/np.mean(drrs_standard-np.mean(drrs_standard))

aux7 = (drrs_standard-np.mean(drrs_standard))*y + np.mean(drrs_standard)

"""
