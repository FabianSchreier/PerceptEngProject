# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import os
import tarfile
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List, Any, Dict, Tuple, BinaryIO, Union

import h5py
import numpy as np
from PIL import Image

from keras.utils import Sequence
from nptyping import NDArray
from numpy.random._generator import default_rng


class DataSource (ABC):
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @abstractmethod
    def get(self, file: str) -> BinaryIO:
        raise NotImplementedError()

    def read(self, file: str) -> bytes:
        with self.get(file) as f:
            return f.read()

    def close(self):
        pass

    def load_files_index(self) -> NDArray[(Any, 3), np.str]:
        array_file = BytesIO()
        array_file.write(self.read('index.npy'))
        array_file.seek(0)
        index = np.load(array_file)

        return index

    def load_files_index_with_source_name(self) -> NDArray[(Any, 4), np.str]:
        idx = self.load_files_index()

        source_name_array = np.full(shape=(idx.shape[0], 1), dtype=idx.dtype, fill_value=self.name)

        augm_idx = np.append(idx, source_name_array, axis=1)
        return augm_idx


class FolderDataSource(DataSource):

    def __init__(self, folder: str):
        self.folder = folder

    @property
    def name(self):
        return os.path.basename(self.folder).lower()

    def get(self, file: str) -> BinaryIO:
        path = os.path.join(self.folder, file)
        if not os.path.exists(path) or not os.path.isfile(path):
            raise FileNotFoundError()
        return open(path, 'rb')


class Hdf5FolderDataSource(DataSource):
    def __init__(self, folder: str):
        self.folder = folder

    @property
    def name(self):
        return os.path.basename(self.folder).lower()

    def get(self, file: str) -> BinaryIO:
        path = os.path.join(self.folder, file)
        if not os.path.exists(path) or not os.path.isfile(path):
            raise FileNotFoundError()
        return open(path, 'rb')


class TarArchiveDataSource(DataSource):
    def __init__(self, archive_file: str, root_folder: str = ''):
        self.archive_file = archive_file
        self.archive = tarfile.open(archive_file, 'r:*')
        self.root_folder = root_folder

    @property
    def name(self):
        file,_ = os.path.splitext(os.path.basename(self.archive_file))
        return file.lower()

    def get(self, file: str) -> BinaryIO:
        f = self.archive.extractfile(os.path.join(self.root_folder, file))
        return f

    def close(self):
        self.archive.close()


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

    sub_set_indx = np.sort(rng.choice(input_indx, size=sub_set_size, replace=False))
    assert len(sub_set_indx) == sub_set_size

    remaining_set_indx = np.delete(input_indx, sub_set_indx)
    assert len(remaining_set_indx) == remaining_set_size

    sub_set = input[sub_set_indx]
    remaining_set = input[remaining_set_indx]

    return sub_set, remaining_set


def load_images(file_entries, input_size, ground_truth_size, sources: Union[DataSource, Dict[str, DataSource]], X=None, y=None, third_channel: str = None):
    # Initialization
    print('Loading data batch of size %d with input size %s and output size %s' % (file_entries.shape[0], input_size, ground_truth_size), flush=True)

    if not isinstance(sources, dict):
        sources = {sources.name: sources}

    if X is None:
        # noinspection PyTypeChecker
        X = np.zeros((file_entries.shape[0], *input_size, 2 if third_channel is None else 3),
                     dtype=np.uint8)  # type: NDArray[(Any, Any, Any, 2), np.uint8]
    if y is None:
        # noinspection PyTypeChecker
        y = np.zeros((file_entries.shape[0], *ground_truth_size, 1),
                     dtype=np.uint8)  # type: NDArray[(Any, Any, Any, 1), np.uint8]

    # Generate data
    for i, tpl in enumerate(file_entries):
        if len(tpl) > 3:
            (image_file_name, scanpath_file_name, fixation_file_name, obs_id, source_name) = tpl
        else:
            (image_file_name, scanpath_file_name, fixation_file_name, obs_id) = tpl
            source_name = next(iter(sources.keys()))    # type: str

        source = sources[source_name]

        image_file = Image.open(source.get(image_file_name))
        scanpath_file = Image.open(source.get(scanpath_file_name))
        fixation_file = Image.open(source.get(fixation_file_name))

        # noinspection PyTypeChecker
        img_data = np.asarray(image_file)           # type: NDArray[(Any, Any), np.uint8]
        # noinspection PyTypeChecker
        scanpath_heatmap = np.asarray(scanpath_file)   # type: NDArray[(Any, Any), np.uint8]
        # noinspection PyTypeChecker
        target_heatmap = np.asarray(fixation_file)     # type: NDArray[(Any, Any), np.uint8]

        assert len(img_data.shape) == 2
        assert img_data.shape == input_size
        assert len(scanpath_heatmap.shape) == 2
        assert scanpath_heatmap.shape == input_size
        assert len(target_heatmap.shape) == 2
        assert target_heatmap.shape == ground_truth_size

        # Store sample
        X[i,:,:,0] = img_data
        X[i,:,:,1] = scanpath_heatmap
        if third_channel == 'image':
            X[i,:,:,2] = img_data
        elif third_channel == 'fixation':
            X[i,:,:,2] = scanpath_heatmap

        # Store class
        y[i,:,:,0] = target_heatmap

    return X, y


class AugmentedDataFolderGenerator(Sequence):
    def __init__(self, files_index: NDArray[(Any, 3), np.str], source: DataSource, batch_size: int = 32, input_size=(480, 640), ground_truth_size=(480, 640), grayscale: bool = True, shuffle: bool = True):
        self.source = source
        self.batch_size = batch_size
        self.input_size = input_size
        self.ground_truth_size = ground_truth_size
        self.grayscale = grayscale
        self.shuffle = shuffle

        self._files_index = files_index

        self._order_indices = np.arange(0, self._files_index.shape[0])
        self._input, self._ground_truth = load_images(self._files_index, self.input_size, self.ground_truth_size, self.source)

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._order_indices)

    def __get(self, index_slice):
        indices = self._order_indices[index_slice]
        X = self._input[indices, :, :, 0:2]
        y = self._ground_truth[indices, :, :, -1]
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
        return int(np.floor(len(self._files_index) / self.batch_size))
