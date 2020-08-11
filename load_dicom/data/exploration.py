import sys
import numpy as np
import os.path
import pickle

#from data.dicom import DicomInterface
#from data.util import get_dirs



def calculate_crop_dims(labels):
    max_r = 0
    max_c = 0
    max_s = 0
    for lab in labels:
        r,c,s = get_roi_boundaries(lab)
        max_r = max_r if (r[1] - r[0]) < max_r else r[1] - r[0]
        max_c = max_c if (c[1] - c[0]) < max_c else c[1] - c[0]
        max_s = max_s if (s[1] - s[0]) < max_s else s[1] - s[0]

    return max_r, max_c, max_s


def calculate_crop_dimensions(datasets):
    labels = []
    for dataset in datasets:
        for case in dataset.keys():
            labels.append(dataset[case]['lab'][0,:,:,:,0])
    r, c, s = calculate_crop_dims(labels)
    r_2,c_2,s_2 = ((2**(r-1).bit_length(),2**(c-1).bit_length(),2**(s-1).bit_length()))
    return (r,c,s),(r_2,c_2,s_2)


def get_roi_boundaries(label_volume):

    rows = [sys.maxsize,-sys.maxsize]
    columns = [sys.maxsize,-sys.maxsize]
    slices = [sys.maxsize,-sys.maxsize]

    for i in range(0,np.shape(label_volume)[2]):

        if 1 in label_volume[:,:,i]:
            slices[0] = min(slices[0],i)
            slices[1] = max(slices[1],i)

    for i in range(0,np.shape(label_volume)[0]):
        if 1 in label_volume[i,:,:]:
            rows[0] =  min(rows[0],i)
            rows[1] = max(rows[1],i)

    for i in range(0,np.shape(label_volume)[1]):
        if 1 in label_volume[:,i,:]:
            columns[0] = min(columns[0],i)
            columns[1] = max(columns[1],i)

    return rows,columns,slices


def roi_center(label_volume):
    b_r, b_c, b_s = get_roi_boundaries(label_volume)
    r = (b_r[1] + b_r[0]) // 2
    c = (b_c[1] + b_c[0]) // 2
    s = (b_s[0] + b_s[1]) // 2

    return (r,c,s)


def extract_roi_patch(img, lab, dims):
    (r, c, s) = roi_center(lab)
    r1 = r - (dims[0] // 2)
    r2 = r + (dims[0] // 2)
    c1 = c - (dims[1] // 2)
    c2 = c + (dims[1] // 2)
    s1 = s - (dims[2] // 2)
    s2 = s + (dims[2] // 2)

    return img[r1:r2,c1:c2,s1:s2], lab[r1:r2,c1:c2,s1:s2], (r,c,s)


def insert_patch(volume, patch, center):
    dims = patch.shape
    r1 = center[0] - (dims[0] // 2)
    r2 = center[0] + (dims[0] // 2)
    c1 = center[1] - (dims[1] // 2)
    c2 = center[1] + (dims[1] // 2)
    s1 = center[2] - (dims[2] // 2)
    s2 = center[2] + (dims[2] // 2)
    volume[r1:r2,c1:c2,s1:s2] =  patch
    return volume


def calculate_oar_patch(rows,columns,slices,dim=96,sdim=96):
    c_rows = int((rows[0] + rows[1])/2)
    c_columns = int((columns[0] + columns[1])/2)
    c_slices =  int((slices[0] + slices[1])/2)

    min_dim = max(rows[1]-rows[0],columns[1]-columns[0],slices[1]-slices[0])

    p_rows = [int(c_rows-dim/2),int(c_rows+dim/2)]
    p_columns = [int(c_columns-dim/2),int(c_columns+dim/2)]
    p_slices = [int(c_slices-sdim/2),int(c_slices+sdim/2)]

    patch = {}
    patch['rows'] = p_rows
    patch['columns'] = p_columns
    patch['slices'] = p_slices
    return patch