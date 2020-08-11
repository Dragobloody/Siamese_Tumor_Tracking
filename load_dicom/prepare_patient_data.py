import dicom_to_numpy as dtn
import numpy as np
from MedImgSeg.dicom import load_dataset


def prepare_p_dict(data_directory, roi_tags = [['tumor'], ['marker']], not_roi_tags = [[], []]):
    # get files locations
    p_dict = dtn.get_p_dict(data_directory)    
    # get label names
    for i in range(len(roi_tags)):
        p_dict = dtn.find_roi(p_dict, roi_tags[i], not_roi_tags[i])
    
    return p_dict


def replace_marker(img, marker):
    img[marker == 1] = np.random.normal(200, 30, img[marker == 1].shape[0])
    return img


def replace_patient_markers(patient):
    for scan in patient.keys():
        keys = patient[scan].keys()
        markers = [s for s in keys if 'marker' in s.lower()]
        for marker in markers:
            patient[scan]['img'] = replace_marker(patient[scan]['img'], patient[scan][marker])
            
    return patient


def generate_patient_data(save_dir, dataset):
    count = 0
    for patient in dataset.keys():          
        patient_data = dtn.load_patient_cases(dataset, patient)        
                
        # get imgs and labs
        imgs = patient_data['img']            
        keys = patient_data.keys()
        tumor = [s for s in keys if 'tv' in s.lower()]            
        labs = [patient_data[tumor[i]] for i in range(len(tumor))]  
        labs = np.concatenate(labs, axis = 0)
                        
        # save data
        save_data_path = save_dir + patient + '/original_data/'
        dtn.save_data(save_data_path, 'imgs', imgs)
        dtn.save_data(save_data_path, 'labs', labs)
            
        count = count + 1
        print(count)
            
            
            
         
            
            
