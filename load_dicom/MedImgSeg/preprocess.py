import sys
sys.path.append('D:/MasterAIThesis/code/load_dicom')
sys.path.append('D:/MasterAIThesis/code/load_dicom/data')

import numpy as np
from numpy.random import choice
import cv2
#from data.data import MedImgExample
from scipy.signal import medfilt

from exploration import roi_center


def median_filter(imgs):
    imgs_filt = []
    for i in range(len(imgs)):
        imgs_filt.append(medfilt(imgs[i]))
    return imgs_filt


def crop_set_around_roi(imgs,labs, info, dims):
    imgs_crop = []
    labs_crop = []
    centers = []
    for i in range(len(imgs)):
        img_crop, lab_crop, center = crop_around_roi(imgs[i],labs[i],dims)
        imgs_crop.append(img_crop)
        labs_crop.append(lab_crop)
        info[i]['roi_center'] = center
    return imgs_crop,labs_crop, info


def window_set_HU(imgs, center, window):
    imgs_window = []
    for i in range(len(imgs)):
        imgs_window.append(HU_window(imgs[i], center, window))
    return np.array(imgs_window)


def normalize_set_2D(imgs):
    imgs_norm = []
    for i in range(len(imgs)):
        imgs_norm.append(cv2.normalize(imgs[i][:,:,0].astype('float32'), np.zeros(imgs[i].shape),
            alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).reshape(imgs[i].shape))
    return np.array(imgs_norm)


def normalize_set_3D(imgs):   
    for i in range(imgs.shape[0]):
        imgs[i] = cv2.normalize(imgs[i,:,:,:].astype('float32'), np.zeros(imgs[i].shape),
            alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).reshape(imgs[i].shape)
    return imgs


def list_examples(dataset,cases):
    dataset_list = {}
    dataset_list['imgs'] = []
    dataset_list['labs'] = []
    dataset_list['info'] = []
    for case in list(dataset.keys()):
        if case in cases:
            dataset_list['imgs'].append(dataset[case]['img'][0])
            dataset_list['labs'].append(dataset[case]['lab'][0])
            dataset_list['info'].append(dataset[case]['dir'])    
    return dataset_list


def crop_around_roi(img, lab, dims):
    (r, c, s) = roi_center(lab)
    r1 = r - (dims[0] // 2)
    r2 = r + (dims[0] // 2)
    c1 = c - (dims[1] // 2)
    c2 = c + (dims[1] // 2)
    s1 = s - (dims[2] // 2)
    s2 = s + (dims[2] // 2)
    if s2 > img.shape[2]:
        s1 = img.shape[2] - dims[2]
        s2 = img.shape[2]

    return img[r1:r2,c1:c2,s1:s2], lab[r1:r2,c1:c2,s1:s2], (r,c,s)


def crop_dataset_around_roi(imgs,labs,dims):
    imgs_crop = []
    labs_crop = []
    for i in range(len(labs)):
        img_crop,lab_crop,_ = crop_around_roi(imgs[i],labs[i],dims)
        imgs_crop.append(img_crop)
        labs_crop.append(lab_crop)
    return imgs_crop, labs_crop


def HU_window(img, center, window):
    max_HU = center + window // 2
    min_HU = center - window // 2
    img[img < min_HU] = min_HU 
    img[img > max_HU] = max_HU
    return img


def preprocess_data(imgs, labs, cases, data_opt):
    #assert sub options
    centers = []
    if 'crop' in data_opt:
        for i in range(len(imgs)):
            imgs[i], labs[i], center = crop_around_roi(imgs[i],labs[i], data_opt['dims'])
            cases[i]['roi_center'] = center

    if 'window' in data_opt:
        for i in range(len(imgs)):
            imgs[i] = HU_window(imgs[i], data_opt['w_center'],
                data_opt['w_width'], cases[i]['intercept'], cases[i]['slope'])
    
    if 'normalize' in data_opt and data_opt['normalize'] == True:
        for i in range(len(imgs)):
            imgs[i] = cv2.normalize(imgs[i], np.zeros(imgs[i].shape),
                alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return imgs, labs, cases


def get_slices(imgs, labs):
    img_slices = []
    lab_slices = []
    for i in range(len(labs)):        
        for j in range(labs[i].shape[2]):
            if 1 in labs[i][:,:,j]:
                img_slices.append(imgs[i][:,:,j])
                lab_slices.append(labs[i][:,:,j])    
    return img_slices, lab_slices


def get_patches(imgs, labs,dims,case_n,empty_n,stride=(1,1,1),roi_volume=1.0):
    a=dims[0]
    b=dims[1]
    c=dims[2]
    
    imgs_patches = []
    labs_patches = []
    for m in range(len(labs)):
        img  = imgs[m]
        lab  = labs[m]
        min_volume = int(np.sum(lab) * roi_volume)    
        case_count = 0
        empty_count = 0
        while case_count < case_n or empty_count < empty_n:
            i = choice(range(0,img.shape[0]-a,stride[0]))
            j = choice(range(0,img.shape[1]-b,stride[1]))
            k = choice(range(0,img.shape[2]-c,stride[2]))               
            if case_count < case_n and np.sum(lab[i:i+a,j:j+b,k:k+c,:])>=min_volume:            
                imgs_patches.append(img[i:i+a,j:j+b,k:k+c,:])
                labs_patches.append(lab[i:i+a,j:j+b,k:k+c,:])
                case_count = case_count + 1
            if empty_count < empty_n and np.sum(lab[i:i+a,j:j+b,k:k+c,:])==0:            
                imgs_patches.append(img[i:i+a,j:j+b,k:k+c,:])
                labs_patches.append(lab[i:i+a,j:j+b,k:k+c,:])
                empty_count = empty_count + 1    
    return imgs_patches, labs_patches


def rescale_data(imgs, labs, max_HU=2500, intercept=24, slope=1):
    opt = dict()
    opt['rescale'] = dict()
    opt['rescale']["max_HU"] = max_HU
    opt['rescale']["intercept"] = intercept
    opt['rescale']["slope"] = slope

    for i in range(len(imgs)):
        imgs[i] = imgs[i].astype('float64') / slope
        imgs[i] = imgs[i] * slope - intercept
        imgs[i][imgs[i] > max_HU] = max_HU
        imgs[i] = imgs[i] / max_HU

    return imgs, labs


def center(dataset):
    x, _, _ = dataset.get_dataset("train")
    mean = np.mean(x)

    opt = dict()
    opt["center"] = dict()
    opt["center"]["mean"] = mean

    examples = dataset.get_examples("train")
    for e in examples:
        x = e.image - mean
        y = e.label
        z = e.parameters

        dataset.delete_examples("train", e.parameters['case'])
        dataset.insert_examples("train", MedImgExample(x, y, z))

    examples = dataset.get_examples("test")
    for e in examples:
        x = e.image - mean
        y = e.label
        z = e.parameters

        dataset.delete_examples("test", e.parameters['case'])
        dataset.insert_examples("test", MedImgExample(x, y, z))

    print("Dataset centered")
    dataset.set_option("preprocess", opt)


def normalize(dataset, stdd=None):
    if stdd is None:
        stdd = np.std(dataset[0], axis=0)
    dataset[0] = dataset[0] / stdd
    return dataset, stdd