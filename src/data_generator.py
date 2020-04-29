# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import os
from typing import List, Any, Dict, Tuple

import h5py
import numpy as np
from PIL import Image

from keras.utils import Sequence
from nptyping import NDArray
from numpy.random._generator import default_rng


def select_subset(input: NDArray, *, num: int = None, ratio: float = None) -> Tuple[NDArray, NDArray]:
    if (num is None and ratio is None) or (num is not None and ratio is not None):
        raise ValueError('Either num or percent must be given.')

    if ratio is not None:
        num = int(input.shape[0] * ratio)

    input_size = input.shape[0]
    sub_set_size = num
    remaining_set_size = input_size - sub_set_size

    input_indx = np.arange(input_size)

    rng = default_rng()

    sub_set_indx =  np.sort(rng.choice(input_indx, size=sub_set_size, replace=False))
    assert len(sub_set_indx) == sub_set_size

    remaining_set_indx = np.delete(input_indx, sub_set_indx)
    assert len(remaining_set_indx) == remaining_set_size

    sub_set = input[sub_set_indx]
    remaining_set = input[remaining_set_indx]

    return sub_set, remaining_set


def load_files_index(folder: str) -> NDArray[(Any, 3), np.str]:
    index_file_name = os.path.join(folder, 'index.npy')
    if not os.path.exists(index_file_name) or not os.path.isfile(index_file_name):
        raise FileNotFoundError()
    index = np.load(index_file_name)
    return index


def load_images(file_entries, image_size, folder, X=None, y=None):
    # Initialization
    print('Loading data batch of size %d' % file_entries.shape[0], flush=True)

    if X is None:
        # noinspection PyTypeChecker
        X = np.empty((file_entries.shape[0], *image_size, 2),
                       dtype=np.float)  # type: NDArray[(Any, Any, Any, 2), np.uint8]
    if y is None:
        # noinspection PyTypeChecker
        y = np.empty((file_entries.shape[0], *image_size, 1),
                       dtype=np.float)  # type: NDArray[(Any, Any, Any, 1), np.uint8]

    # Generate data
    for i, (image_file_name, scanpath_file_name, fixation_file_name) in enumerate(file_entries):

        image_file = Image.open(os.path.join(folder, image_file_name))
        scanpath_file = Image.open(os.path.join(folder, scanpath_file_name))
        fixation_file = Image.open(os.path.join(folder, fixation_file_name))

        # noinspection PyTypeChecker
        img_data = np.asarray(image_file)           # type: NDArray[(Any, Any), np.uint8]
        # noinspection PyTypeChecker
        scanpath_heatmap = np.asarray(scanpath_file)   # type: NDArray[(Any, Any), np.uint8]
        # noinspection PyTypeChecker
        target_heatmap = np.asarray(fixation_file)     # type: NDArray[(Any, Any), np.uint8]

        assert len(img_data.shape) == 2
        assert img_data.shape == image_size
        assert len(scanpath_heatmap.shape) == 2
        assert scanpath_heatmap.shape == image_size
        assert len(target_heatmap.shape) == 2
        assert target_heatmap.shape == image_size

        # Store sample
        X[i,:,:,0] = img_data
        X[i,:,:,1] = scanpath_heatmap

        # Store class
        y[i,:,:,-1] = target_heatmap

    return X, y


class AugmentedDataFolderGenerator(Sequence):
    def __init__(self, files_index: NDArray[(Any, 3), np.str], folder: str, batch_size: int = 32, image_size= (480, 640), grayscale: bool = True, shuffle: bool = True):
        self.folder = folder
        self.batch_size = batch_size
        self.image_size = image_size
        self.grayscale = grayscale
        self.shuffle = shuffle

        self._files = files_index

        self._images = self.__data_generation(self._files)


        self.on_epoch_end()


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._images)

    def __data_generation(self, file_entries) -> NDArray[(Any, Any, Any, 3), np.uint8]:
        # noinspection PyTypeChecker
        res = np.empty((file_entries.shape[0], *self.image_size, 3), dtype=np.float)  # type: NDArray[(Any, Any, Any, 3), np.uint8]

        load_images(file_entries, self.image_size, self.folder, res[:, :, :, 0:2], res[:, :, :, 2])

        return res

    def __get(self, index_slice):
        X = self._images[index_slice, :, :, 0:2]
        y = self._images[index_slice, :, :, -1]
        y.shape = (*y.shape, 1)     # Add extra dim (so the Conv output layer is happy)
        return X, y

    def get_all_items(self):
        X, y = self.__get(slice(None))

        X = X.copy()
        y = y.copy()
        return X, y

    def __getitem__(self, index):
        #files = self._files_order[index*self.batch_size:(index+1)*self.batch_size]
        #X, y = self.__data_generation(files)

        '''X = self._images[index*self.batch_size:(index+1)*self.batch_size, :, :, 0:2]
        y = self._images[index*self.batch_size:(index+1)*self.batch_size, :, :, -1]
        y.shape = (*y.shape, 1)     # Add extra dim (so the Conv output layer is happy)

        return X, y
        '''

        X, y = self.__get(slice(index*self.batch_size, (index+1)*self.batch_size))
        return X, y

    def __len__(self):
        return int(np.floor(len(self._files) / self.batch_size))
