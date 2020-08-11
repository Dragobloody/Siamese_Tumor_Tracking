
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

#from data.config import data_path
#from data.config import datasets


def get_dirs(dir,contains=""):
    dirList = []
    for dirName, subdirList, filelist in os.walk(dir):
        if contains in dirName:
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


def create_dataset_file(data_path, roi):
    f_name = data_path + roi + time.strftime("_%Y%m%d-%H%M") + ".hdf5"
    data_opt = dict()
    data_opt['roi'] = roi
    with open_dataset(f_name, "w") as f:
        f.create_dataset("data_opt", data=json.dumps(data_opt))
    return f_name


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