import pydicom
import numpy as np
from data.util import get_file_list
from data.util import get_dirs
from PIL import Image, ImageDraw
import h5py
import os
import json
from sklearn.model_selection import train_test_split


class InvalidROI(Exception):
    pass


class DicomInterface(object):
    """docstring for DicomInterface"""
    def __init__(self, dicom_dir, rs_file = None):
        self.dicom_dir = dicom_dir
        self.ct_files = get_file_list(dicom_dir, "CT", ".dcm")
        if rs_file is None:
            self.rs_file = get_file_list(dicom_dir, "RS", ".dcm")[0]
        else:
            self.rs_file = rs_file
        self.target_rois = []

    def get_patient_id(self):
        return int(pydicom.read_file(self.rs_file).PatientID)

    def get_image_parameters(self, ct_files):
        slices = []
        for ct_file in ct_files:
            ct = pydicom.read_file(ct_file)
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
        patient_dir = self.dicom_dir

        ct_file_list = get_file_list(patient_dir, "CT", ".dcm")
        slices, center, spacing, dims = self.get_image_parameters(ct_file_list)

        roi_index = self.get_structures(ds_file)[roi]['index']
        contour_data = self.get_contour_data(ds_file, roi_index)
        slice_contours = self.get_slice_contours(contour_data, slices)

        label_volume = np.zeros(dims, dtype=np.dtype('uint8'))
        for k in range(0, len(slice_contours.keys())):
            contour = slice_contours[slices[k]]
            slice_label = self.draw_slice_label(contour, center, spacing, dims)
            label_volume[:, :, k] = slice_label

        if slices[0] > slices[1]:
            label_volume = label_volume[:, :, ::-1]
        label_volume = label_volume.reshape((1,)+dims+(1,))
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
            for i in range(0, 512):
                for j in range(0, 512):
                    slice_label[i, j] = img.getpixel((j, i))
        return slice_label

    def get_image_volume(self):
        ct_file_list = self.ct_files
        slices, _, _, dims = self.get_image_parameters(ct_file_list)
        image_volume = np.zeros(dims)
        for ct_file in ct_file_list:
            ct = pydicom.read_file(ct_file)
            image_volume[:, :, ct_file_list.index(ct_file)] = ct.pixel_array

        if slices[0] > slices[1]:
            image_volume = image_volume[:, :, ::-1]
        image_volume = image_volume * ct.RescaleSlope + ct.RescaleIntercept
        image_volume = image_volume.reshape((1,)+dims+(1,))
        return image_volume


def get_patient_dict(rs_file_list):
    patient_dict = {}
    for rs_file in rs_file_list:
        rs = pydicom.read_file(rs_file)
        patient_dict[int(rs.PatientID)] = rs_file
    return patient_dict

def parse_directory(directory):
    dir_list = get_dirs(directory)
    p_dict = {}
    for directory in dir_list:
        p = DicomInterface(directory)
        p_dict[p.get_patient_id()] = p
    return p_dict

def extract_dataset(p_dict, cases=None):
    dataset = {}
    error = {}
    if cases is None:
        cases = list(p_dict.keys())
    for case in cases:
        try:
            label = p_dict[case].get_label_volume()
            dataset[case] = {}
            dataset[case]['img'] = p_dict[case].get_image_volume()
            dataset[case]['lab'] = label
            dataset[case]['dir'] = p_dict[case].dicom_dir
        except:
            error[case] = "error"
    return dataset, error

def discover_names(p_dict, roi_tags, not_roi_tags=None):
    tags = []
    for case in p_dict.keys():
        structures = p_dict[case].get_structures()
        for s in structures:
            flag = False
            for tag in roi_tags:
                if tag in s:
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
                tags.append(s)
    return set(tags)

def assign_roi_label(p_dict, roi_candidates):
    del_list = []
    m_count = 0
    for patient in p_dict.keys():
        p_dict[patient].set_target_rois(roi_candidates)
        if len(p_dict[patient].target_rois) < 1:
            del_list.append(patient)
        if len(p_dict[patient].target_rois) > 1:
            m_count = m_count + 1
    for patient in del_list:
        del p_dict[patient]
    return len(del_list), m_count, p_dict

def save_dataset(dataset, data_dir, base_name):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for case in dataset.keys():
        f = h5py.File(data_dir + base_name + str(case),"w")
        f.attrs['case'] = case
        f.create_dataset("img", data=dataset[case]['img'])
        f.create_dataset("lab", data=dataset[case]['lab'])
        f.create_dataset("dir", data=json.dumps(dataset[case]['dir']))
        f.close()
    print("Dataset saved.")

def load_dataset(data_dir, base_name):
    dataset = {}
    for file in get_file_list(data_dir,base_name,""):
        f = h5py.File(file,"r")
        case = f.attrs['case']
        dataset[case] = {}
        dataset[case]['img'] = f['img']
        dataset[case]['lab'] = f['lab']
        dataset[case]['dir'] = json.loads(f['dir'].value)
    return dataset

def split_dataset(dataset, test_split):
    train_cases, test_cases = train_test_split(list(dataset.keys()),test_size=test_split)
    train_dataset = {}
    for case in train_cases:
        train_dataset[case] = dataset[case]
    test_dataset = {}
    for case in test_cases:
        test_dataset[case] = dataset[case]
    return train_dataset, test_dataset