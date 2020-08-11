from pathlib import Path

import pydicom
import numpy as np
import h5py
import os
import json
import scipy
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw

from MedImgSeg.util import get_file_list
from MedImgSeg.util import get_dirs


class InvalidROI(Exception):
    pass


class DicomInterface(object):
    """docstring for DicomInterface"""

    '''TODO: The way this interface parses dicom paths really need some cleaning up''' 
    
    def __init__(self, dicom_dir, rs_file=None, new_style=True, rs_dir=None):        
        self.dicom_dir = dicom_dir
        if new_style:
            self.ct_files = sorted(path.as_posix() for path in Path(dicom_dir).iterdir())
        else:
            self.ct_files = get_file_list(dicom_dir, "CT", ".dcm")
        if rs_file is None:
            if new_style:
                assert rs_dir is not None
                self.rs_file = next(iter(Path(rs_dir).iterdir())).as_posix()
            else:
                self.rs_file = get_file_list(dicom_dir, "RS", ".dcm")[0]
        else:
            self.rs_file = rs_file
        self.target_rois = []  

    def get_patient_id(self):
        return pydicom.read_file(self.rs_file).PatientID
            
    def get_image_parameters(self, ct_files):
        '''Currently this should return the parameters of the last one'''
        slices = []
        scans = [pydicom.read_file(s) for s in ct_files]
        scans.sort(key = lambda x: int(x.InstanceNumber))
        for ct in scans:        
            slices.append(ct.ImagePositionPatient[2])            
        
        center = (ct.ImagePositionPatient[0],ct.ImagePositionPatient[1])
        center = np.around(center, decimals=1)
        
        spacing = (float(ct.PixelSpacing[0]) ,float(ct.PixelSpacing[1]), float(ct.SliceThickness))
        dims = (int(ct.Rows), int(ct.Columns), len(ct_files))
        
        return slices, center, spacing, dims

    def set_target_rois(self,roi_candidates):
        for roi_name in roi_candidates:            
            if self.check_contour_data(roi_name): 
                self.target_rois.append(roi_name)                
        return self.target_rois

    def check_contour_data(self,roi):
        try:
            roi_index = self.get_structures(self.rs_file)[roi]['index']
            contour_data = self.get_contour_data(self.rs_file,roi_index)            
            assert len(contour_data) > 0
            return True
        except:
            return False
    
    def get_slice_contours(self, contour_data, slices):
        slice_contours = {k: [] for k in slices}
        for i in range(0, len(contour_data)):            
            slice_contours[contour_data[i][2]].append(contour_data[i])
        return slice_contours
    
    def get_structures(self, rs_file=None):
        if rs_file is None:
            rs_file = self.rs_file        
        rs = pydicom.read_file(rs_file)
        structures = {}        
        for i in range(len(rs.StructureSetROISequence)):
            structures[rs.StructureSetROISequence[i].ROIName] = {}
            structures[rs.StructureSetROISequence[i].ROIName]['index'] = i
        return structures



    def get_contour_data(self, rs_file, roi_index):
        rs = pydicom.read_file(rs_file)
        contour_seq = rs.ROIContourSequence[roi_index].ContourSequence        
        roi_contour_data = []
        for slice_contour in contour_seq:            
            slice_contour_data = slice_contour.ContourData           
            for i in range(len(slice_contour_data)):
                slice_contour_data[i] = slice_contour_data[i]
            roi_contour_data.append(slice_contour_data)
        return roi_contour_data
    
   
    def get_label_volume(self, roi=None, target_index=0):
        if roi == None:
            roi = self.target_rois[target_index]        

        ds_file = self.rs_file
        patient_dir =self.dicom_dir

        ct_file_list = self.ct_files
        slices, center, spacing, dims = self.get_image_parameters(ct_file_list)    
        
        roi_index = self.get_structures(ds_file)[roi]['index']
        contour_data = self.get_contour_data(ds_file, roi_index)
        slice_contours = self.get_slice_contours(contour_data, slices)

        label_volume = np.zeros(dims, dtype=np.dtype('uint8'))         
        for k, sl  in enumerate(slice_contours):             
            contour = slice_contours[sl]            
            slice_label = self.draw_slice_label(contour, center, spacing, dims)
            label_volume[:, :, k] = slice_label
        
        if slices[0] > slices[1]:            
            label_volume = label_volume[:, :, ::-1]
        label_volume = label_volume.reshape(dims)
        return label_volume

    def draw_slice_label(self, contour, center, spacing, dims):
        img = Image.new("1", (dims[0], dims[1]))
        draw = ImageDraw.Draw(img)
        slice_label = np.zeros([dims[0], dims[1]])
        for c in contour:        
            x = [c[i] for i in range(0, len(c)) if i % 3 == 0]
            y = [c[i] for i in range(0, len(c)) if i % 3 == 1]
            x.append(x[0])
            y.append(y[0])        
            poly = [(int((x - center[0]) / spacing[0]), int((y - center[1]) / spacing[0])) for x, y in zip(x, y)]
            draw.polygon(poly, fill=1, outline=1)
            for i in range(0, dims[1]):
                for j in range(0, dims[0]):
                    slice_label[i, j] = img.getpixel((j, i))
        return slice_label

    def get_image_volume(self):
        ct_file_list = self.ct_files
        slices, _, _, dims = self.get_image_parameters(ct_file_list)
        image_volume = np.zeros(dims)
        idx = []
        for ct_file in ct_file_list:  
            idx.append(ct_file_list.index(ct_file))
            ct = pydicom.read_file(ct_file)
            image_volume[:, :, ct.InstanceNumber-1] = ct.pixel_array        

        if slices[0] > slices[1]:            
            image_volume = image_volume[:, :, ::-1]
        image_volume = image_volume * ct.RescaleSlope + ct.RescaleIntercept    
        image_volume = image_volume.reshape(dims)
        return image_volume


def get_patient_dict(rs_file_list):
    patient_dict = {}
    for rs_file in rs_file_list:    
        rs = pydicom.read_file(rs_file)
        patient_dict[int(rs.PatientID)] = rs_file
    return patient_dict


def parse_directory(directory, all_scans = False): 
    p_dict = {}
    for patient_id, respiratory_phases in directory.items():            
            p_dict[patient_id] = {}
            for i, respiratory_phase in enumerate(respiratory_phases):                   
                    ct_dir, rs_dir = respiratory_phase['CT'], respiratory_phase['RT']
                    p = DicomInterface(dicom_dir=ct_dir, rs_dir=rs_dir)
                    p_dict[patient_id][i] = p
   
    return p_dict


def discover_names(p_dict, roi_tags, not_roi_tags=None):
    '''This method searches for a specific set of indicator strings within each found ROI tag to find a specific ROI.
    '''
    # roi_tags should be a list of strings
    tags = set()
    roi_tags = list(map(lambda x: x.lower(), roi_tags))    
    for patient in p_dict.keys():                   
            for j, phase in p_dict[patient].items():               
                structures = phase.get_structures()                                           
                for s in structures:
                    flag = False
                    for tag in roi_tags:                        
                        if tag in s.lower():
                            flag = True
                        else:
                            flag = False
                            break
        
                    for not_tag in not_roi_tags:
                        if not_tag in s:
                            flag = False
                        else:
                            continue                        
                    
                    if flag:
                        tags.add(s)                                             
                        
    return tags


def assign_roi_label(p_dict, roi_candidates):
    del_list = []
    m_count = 0    
    s_count = 0
    for patient in p_dict.keys():                 
            for i, phase in p_dict[patient].items(): 
                p_dict[patient][i].set_target_rois(roi_candidates)   
                if len( p_dict[patient][i].target_rois) == 1:
                    s_count = s_count + 1
                if len( p_dict[patient][i].target_rois) > 1:
                    m_count = m_count + 1
                if len( p_dict[patient][i].target_rois) < 1:
                    del_list.append(patient + '_' + str(i))
  
    return len(del_list), m_count, s_count, p_dict



def extract_cases(p_dict, data_dir, base_name):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir) 
    
    patients = list(p_dict.keys())
    
    error = {}
    count = 0
    for patient in patients:       
        patient_dir = data_dir + patient + '/'
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        scan_dir = patient_dir + 'original_data/'
        if not os.path.exists(scan_dir):
            os.makedirs(scan_dir)   
            
           
        for i, phase in p_dict[patient].items(): 
                _, _, spacing, _ =  p_dict[patient][i].get_image_parameters(p_dict[patient][i].ct_files)                
                count = count + 1                
                if 0 in spacing:
                    continue
                #new_spacing = (1.0, 1.0, 1.0)
                
                labs = {}
                img = p_dict[patient][i].get_image_volume()
                # applying Hounsfield Unit window
                #window_center = -300
                #window_width = 1400
                #img = HU_window(img, window_center, window_width)                
                
                for idx, label in enumerate(p_dict[patient][i].target_rois):
                    lab = p_dict[patient][i].get_label_volume(target_index = idx)                     
                    lab[lab > 1] = 1
                    
                    # replace markers
                    if 'marker' in label.lower():
                        img = replace_marker(img, lab)
                    #elif 'tumor' in label.lower():
                    #    lab, _ = resample(lab, spacing, new_spacing)
                    #    labs['tumor'] =  lab 
                    else:
                        labs[label] = lab
                
                #img, new_spacing = resample(img, spacing, new_spacing)                
                
                f = h5py.File(scan_dir + base_name + str(i), "w")     
                f.create_dataset("img", data=img, dtype=np.int16)
                for key, value in labs.items():
                    f.create_dataset(key, data=value)                
                f.create_dataset("spacing", data=spacing)    
                #f.create_dataset("new_spacing", data=new_spacing)    
                f.close()    
                print(count)
                
               
    return count, error



def replace_marker(img, marker):
    img[marker == 1] = np.random.normal(200, 30, img[marker == 1].shape[0])
    return img

def HU_window(img, center, window):
    max_HU = center + window // 2
    min_HU = center - window // 2
    img[img < min_HU] = min_HU 
    img[img > max_HU] = max_HU
    return img

def load_dataset(data_dir, base_name):
    dataset = {}
    patients = list(Path(data_dir).iterdir())
    for p in patients:
        patient = p.name
        dataset[patient] = {}      
                       
        for i, file in enumerate(get_file_list(data_dir + patient + '/original_data/', base_name, "")):
           
            f = h5py.File(file,"r")                
            dataset[patient][base_name + str(i)] = {}
            for key in f.keys():
                dataset[patient][base_name + str(i)][key] = f[key]                
                       
    return dataset


def resample(data, spacing, new_spacing = [1.0, 1.0, 1.0]):
    # Determine current pixel spacing    
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = data.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / data.shape
    new_spacing = spacing / real_resize_factor
    
    data = scipy.ndimage.interpolation.zoom(data, real_resize_factor)
    
    return data, new_spacing
