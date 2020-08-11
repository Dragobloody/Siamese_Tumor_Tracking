from sklearn.model_selection import train_test_split
import h5py
import sys
import numpy as np

from data.dicom import DicomInterface
from data.dicom import InvalidROI


from data.util import get_dirs
from data.util import create_dataset_file
from data.util import insert_example
from data.util import delete_example
from data.util import list_cases
from data.util import get_example
from data.util import get_dataset_options
from data.util import set_dataset_options
from data.util import create_dataset

from data.config import data_path


def generate_bound_box_output_2D(slice_label):
    row = [sys.maxsize,-sys.maxsize]
    column = [sys.maxsize,-sys.maxsize]

    for i in range(0,slice_label.shape[0]):

        if 1 in slice_label[i,:]:
            column[0] = min(column[0],i)
            column[1] = max(column[1],i)

    for i in range(0,slice_label.shape[1]):
        if 1 in slice_label[:,i]:
            row[0] =  min(row[0],i)
            row[1] = max(row[1],i)

    if row[0] == sys.maxsize:
        box = 0
        x = y = w = h = None
    else:
        box = 1
        x = (row[1] + row[0]) // 2
        y = (column[1] + column[0]) // 2
        w = row[1] - row[0]
        h = column[1] - column[0]
    return [box,x, y, w, h]


def preprocess_training(x_train, y_train, x_dev, y_dev, data_opt, f_set):

    if 'w_center' in data_opt and 'w_width' in data_opt:
        x_train = f_set['window'](x_train,data_opt['w_center'],data_opt['w_width'])
        x_dev = f_set['window'](x_dev,data_opt['w_center'],data_opt['w_width'])

    if 'normalize' in data_opt and data_opt['normalize'] == True:
        x_train =  f_set['normalize'](x_train)
        x_dev =  f_set['normalize'](x_dev)

    if 'center' in data_opt and data_opt['center'] == True:
        mean = data_opt['center_mean'] = np.mean(x_train)
        x_train = x_train - mean
        x_dev = x_dev - mean

    return x_train, y_train, x_dev, y_dev


def load_dataset(f_name,dev_data=False,test_data=False):
    f = h5py.File(f_name,'r')
    x_train = f['x_train'][:]
    y_train = f['y_train'][:]
    if dev_data:
        x_dev = f['x_dev'][:]
        y_dev = f['y_dev'][:]
    else:
        x_dev = None
        y_dev = None
    if test_data:
        x_test = f['x_test'][:]
        y_test = f['y_test'][:]
    else:
        x_test = None
        y_test = None
    f.close()

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def reshape_data(data, shape):

    result = []
    for d in data:
        result.append(d.reshape(shape))
    return result


class MedImgData:

    def __init__(self, f_name):
        self.f_name = f_name

    def insert_examples(self, dataset, examples):
        if type(examples) is not list:
            examples = [examples]
        for e in examples:
            insert_example(self.f_name, dataset, e.image, e.label, e.parameters)

    def delete_examples(self, dataset, cases):
        if type(cases) is not list:
            cases = [cases]
        for c in cases:
            delete_example(self.f_name, dataset, c)

    def get_examples(self, dataset, cases=None):
        if cases is None:
            cases = list_cases(self.f_name, dataset)
        if type(cases) is not list:
            cases = [cases]
        examples = []
        for c in cases:
            img, lab, par = get_example(self.f_name, dataset, c)
            examples.append(MedImgExample(img, lab, par))
        return examples

    def list_cases(self, dataset):
        return list_cases(self.f_name, dataset)

    def split_dataset(self, test_size=0.1, dev_size=None):
        opt = dict()
        opt["test_split"] = test_size
        opt["dev_split"] = dev_size

        f_name = create_dataset_file(data_path, "PAROTIS_RE")
        roi_dataset = MedImgData(f_name)

        cases = list_cases(self.f_name, "orig")
        cases_train, cases_test = train_test_split(cases,
                                                   test_size=test_size)

        if dev_size is not None:
            cases_train, cases_dev = train_test_split(cases_train,
                                                      test_size=dev_size)
            examples_dev = self.get_examples("orig", cases_dev)
            create_dataset(f_name, "dev")
            roi_dataset.insert_examples("dev", examples_dev)

        examples_train = self.get_examples("orig", cases_train)
        create_dataset(f_name, "train")
        roi_dataset.insert_examples("train", examples_train)

        examples_test = self.get_examples("orig", cases_test)
        create_dataset(f_name, "test")
        roi_dataset.insert_examples("test", examples_test)
        self.set_option("sets", opt)

        return roi_dataset

    def get_dataset(self, dataset):
        cases = list_cases(self.f_name, dataset)
        x = []
        y = []
        z = []
        for c in cases:
            img, lab, par = get_example(self.f_name, dataset, c)
            x.append(img)
            y.append(lab)
            z.append(par["case"])

        return x, y, z

    def get_dataset_(self):
        pass

    def set_option(self, opt, value):
        data_opt = get_dataset_options(self.f_name)
        if opt in data_opt:
            data_opt[opt].update(value)
        else:
            data_opt[opt] = value
        set_dataset_options(self.f_name, data_opt)

    def get_options(self):
        return get_dataset_options(self.f_name)


class MedImgExample:
    def __init__(self, parameters):
        self.parameters = parameters

    def get_image_volume(self):
        return DicomImage(self.parameters['directory']).get_image_volume()

    def get_label_volume(self,roi):
        return DicomImage(self.parameters['directory']).get_label_volume(roi)

    @classmethod
    def from_dicom(cls, dicom_dir, roi):
        p = DicomImage(dicom_dir)
        parameters = dict()
        parameters['structures'] = p.structures
        if roi not in parameters['structures']:
            return None
        parameters['origin'] = 'dicom'
        parameters['directory'] = p.directory
        parameters['case'] = p.case
        parameters['patient_id'] = p.patient_id
        parameters['pixel_array_type'] = p.pixel_array_type.name
        parameters['dims'] = p.dims
        parameters['order'] = p.order
        parameters['spacing'] = p.spacing
        parameters['centers'] = p.centers
        parameters['roi'] = roi

        return cls(parameters)


def save_train_data(imgs_train,labs_train, imgs_dev=None, labs_dev=None, data_dir="", name="train_data"):
    f_name = data_dir + "/" + name + time.strftime("_%Y%m%d-%H%M")+".hdf5"
    with open_dataset(f_name, "w") as f:
        x_train, y_train = np.stack(imgs_train), np.stack(labs_train)
        f.create_dataset("x_train", data=x_train)
        f.create_dataset("y_train", data=y_train)
        print("Shape of training data: ", x_train.shape)
        if imgs_dev is not None:
            x_dev, y_dev = np.stack(imgs_dev), np.stack(labs_dev)
            f.create_dataset("x_dev", data=x_dev)
            f.create_dataset("y_dev", data=y_dev)
            print("Shape of dev data: ", x_dev.shape)
    print("Train data saved in", f_name)