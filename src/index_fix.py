# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"


import os
import numpy as np
import datasets
from data_generator import load_files_index


if __name__ == '__main__':
    dataset_folder = os.path.join(datasets.root_folder, 'ProcessedDatasets', 'Cat2000')
    index = load_files_index(dataset_folder)

    wrong_folder = '/scratch/258617/ProcessedDatasets/Cat2000/'
    start_index = len(wrong_folder)

    new_index = np.char.replace(index, wrong_folder, '')

    sorted_index = np.sort(new_index, 0)
    '''
    new_index = np.empty(shape=index.shape, dtype=np.dtype('<U%d' % (87 - start_index)))

    for i, (img, scanpath, fixation) in enumerate(index):
        print('Fixing %d of %d' % (i+1, index.shape[0]))
        rel_img = img[start_index:]
        rel_scanpath = scanpath[start_index:]
        rel_fixation = fixation[start_index:]

        new_index[i, 0] = rel_img
        new_index[i, 1] = rel_scanpath
        new_index[i, 2] = rel_fixation
    '''

    np.save(os.path.join(dataset_folder, 'index.npy'), sorted_index)