import numpy as np
import h5py

from read_dicom import get_patient_paths, extract_CT_and_RT_paths

from MedImgSeg.dicom import parse_directory
from MedImgSeg.dicom import discover_names
from MedImgSeg.dicom import assign_roi_label
from MedImgSeg.dicom import extract_cases


def save_cases(data_dir, base_name, case, img, lab, body, spacing = None):
    f = h5py.File(data_dir + base_name + str(case), "w")
    f.attrs['case'] = case
    f.create_dataset("img", data=img, dtype=np.int16)
    f.create_dataset("lab", data=lab)
    f.create_dataset("body", data=body)
    if spacing is not None:
        f.create_dataset("spacing", data=spacing)    
    f.close()        
       



def get_p_dict(data_directory):
    # Parse our dataset to 'dicom_directory' style
    patient_paths = get_patient_paths(data_directory)
    extracted_scan_paths = extract_CT_and_RT_paths(patient_paths)    
    # Finally, parse to their p_dict in
    p_dict = parse_directory(directory=extracted_scan_paths, all_scans = True)
    
    return p_dict


def find_roi(p_dict, roi_tags, not_roi_tags):
    roi_candidates = discover_names(p_dict, roi_tags, not_roi_tags)
    # assign valid label names to examples in the dataset
    #p_dict = parse_directory(dicom_directory)
    d_count, m_count, s_count, p_dict = assign_roi_label(p_dict, roi_candidates)
    print("Number of cases without ROI candidates contour data: " , d_count)
    print("Number of cases with a single contour: ", s_count)
    print("Number of cases with multiple ROI candidates contour data", m_count)
    return p_dict
   

def create_cases(p_dict, data_directory):
    # extract train set (this may take a while)
    base_name =  "/phase_"
    count, error = extract_cases(p_dict, data_directory, base_name)
    print("Number of cases extracted: ", count)
    print("Number of cases not extracted: ", len(list(error.keys())))    
    

def load_patient_cases(dataset, patient):
    dictionary = {}    
    for key in dataset[patient]['phase_0'].keys():
        dictionary[key] = []
    dict_keys = list(dictionary.keys())
    for phase in dataset[patient].keys():            
        for i, value in enumerate(dataset[patient][phase].values()):
            dictionary[dict_keys[i]].append(np.array(value))
        
    for key in dictionary.keys():
            dictionary[key] = np.stack(dictionary[key], axis = 0)
    
    return dictionary

    

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
