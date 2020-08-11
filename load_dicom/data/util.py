
import sys
from numbers import Number
from collections import Set, Mapping, deque
import math
import h5py
import os
import time
from natsort import natsorted
import json
from contextlib import contextmanager

from data.config import data_path
from data.config import datasets

def get_dirs(dir):
    dirList = []
    for dirName, subdirList, filelist in os.walk(dir):
        dirList.append(dirName)
    dirList = natsorted(dirList)
    return dirList[1:]


def get_file_list(dir, name, extension):
    file_list = []
    for dirName, subdirList, filelist in os.walk(dir):
        for filename in filelist:
            if (extension in filename.lower()) and (name in filename):
                file_list.append(os.path.join(dirName, filename))

    file_list = natsorted(file_list)
    return file_list



def get_dataset_options(f_name):
    with open_dataset(f_name, "r") as f:
        return json.loads(f['data_opt'].value)


def set_dataset_options(f_name, data_opt):
    with open_dataset(f_name, "r+") as f:
        del f['data_opt']
        f.create_dataset("data_opt", data=json.dumps(data_opt))


@contextmanager
def open_dataset(f_name, mode):
    # TODO: check f_name and if file exists and mode is w, throw excpetion
    f = h5py.File(f_name, mode)
    try:
        yield f
    finally:
        f.close()


def create_dataset(f_name, dataset):
    with open_dataset(f_name, "r+") as f:
        if dataset not in f:
            f.create_group(dataset)
            return 1
        return 0


def delete_dataset(f_name, dataset):
    with open_dataset(f_name, "r+") as f:
        del f[dataset]


def create_dataset_file(data_path, roi):
    f_name = data_path + roi + time.strftime("_%Y%m%d-%H%M") + ".hdf5"
    data_opt = dict()
    data_opt['roi'] = roi
    with open_dataset(f_name, "w") as f:
        f.create_dataset("data_opt", data=json.dumps(data_opt))
    return f_name


def insert_example(f_name, dataset, img, lab, info):
    with open_dataset(f_name, "r+") as f:
        case = info['case']
        node = dataset + "/" + case
        if node not in f:
            f.create_group(node)
            f[node].create_group(case)
            f[node].create_dataset('img', data=img)
            f[node].create_dataset('lab', data=lab)
            f[node].create_dataset('par', data=json.dumps(info))
            return 1
        else:
            return 0


def delete_example(f_name, dataset, case):
    with open_dataset(f_name, "r+") as f:
        node = dataset + "/" + case
        if node in f:
            del f[node]
            print("%s deleted from dataset" % node)
        else:
            print("%s not in dataset" % node)


def list_cases(f_name, dataset):
    with open_dataset(f_name, "r") as f:
        cases = []
        for k in f[dataset].keys():
            cases.append(k)
    return cases


def list_datasets(f_name):
    with open_dataset(f_name, "r") as f:
        dataset_list = []
        for k in f.keys():
            if k in datasets:
                dataset_list.append(k)
    return dataset_list


def get_example(f_name, dataset, case):
    node = dataset + "/" + case
    with open_dataset(f_name, "r") as f:
        if node in f:
            img = f[dataset][case]["img"].value
            lab = f[dataset][case]["lab"].value
            par = json.loads(f[dataset][case]["par"].value)
        else:
            print("Example %s not in dataset %s" % (case, dataset))
            return None, None, None

        #shape = list(img.shape)
        #shape.append(1)

    return img, lab, par


def load_dataset_file(oar, data_id, mode):
    f_name = data_path + oar + "/" + str(data_id) + ".hdf5"
    f = h5py.File(f_name, mode)
    return f

def save_dataset(dataset_opt, train_set, dev_set, test_set ):
    data_id =  get_new_data_id(dataset_opt['oar'])
    f_name = data_path + dataset_opt['oar'] + "/" + str(data_id) + ".hdf5"
    f = h5py.File(f_name)

    f.create_group("train")
    f["train"].create_dataset("img", data = train_set[0].reshape(train_set[0].shape + (1,)))
    f["train"].create_dataset("lab", data = train_set[1].reshape(train_set[1].shape + (1,)))
    f["train"].create_dataset("cas", data = train_set[2])
    f["train"].attrs.create("shape", data = train_set[0].shape)

    f.create_group("dev")
    f["dev"].create_dataset("img", data = dev_set[0].reshape(dev_set[0].shape + (1,)))
    f["dev"].create_dataset("lab", data = dev_set[1].reshape(dev_set[1].shape + (1,)))
    f["dev"].create_dataset("cas", data = dev_set[2])
    f["dev"].attrs.create("shape", data = dev_set[0].shape)

    f.create_group("test")
    f["test"].create_dataset("img", data = test_set[0].reshape(test_set[0].shape + (1,)))
    f["test"].create_dataset("lab", data = test_set[1].reshape(test_set[1].shape + (1,)))
    f["test"].create_dataset("cas", data = test_set[2])
    f["test"].attrs.create("shape", data = test_set[0].shape)

    f.attrs.create("data_shape", data=train_set[0].shape[1:])


def load_dataset(oar, data_id):

    x_train, y_train, _ = load_set(oar,data_id, "train")
    x_dev, y_dev, _ = load_set(oar,data_id, "dev")
    x_test, y_test, _ = load_set(oar,data_id, "dev")

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def load_set(oar, data_id, dataset):
    f_name = data_path + oar + "/" + str(data_id) + ".hdf5"

    f = h5py.File(f_name, "r")
    x = f[dataset + "/img"][:]
    y = f[dataset + "/lab"][:]
    c = f[dataset + "/cas"][:]
    f.close()

    return x, y, c


def get_new_data_id(oar):
    pass

def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    try: # Python 2
        zero_depth_bases = (basestring, Number, xrange, bytearray)
        iteritems = 'iteritems'
    except NameError: # Python 3
        zero_depth_bases = (str, bytes, Number, range, bytearray)
        iteritems = 'items'
    def inner(obj, _seen_ids = set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    def convert_size(size_bytes):
       if size_bytes == 0:
           return "0B"
       size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
       i = int(math.floor(math.log(size_bytes, 1024)))
       p = math.pow(1024, i)
       s = round(size_bytes / p, 2)
       return "%s %s" % (s, size_name[i])
    return convert_size(inner(obj_0))