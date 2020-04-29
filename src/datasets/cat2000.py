# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import os
from typing import Any, Iterable, List

import numpy as np
import scipy.io
from PIL import Image
from nptyping import NDArray

from . import datasets_folder, InputData, swap_columns


name = 'Cat2000'

def images_path(datasets_root: str = None) -> str:
    if datasets_root is None:
        datasets_root = datasets_folder
    return os.path.join(datasets_root, 'Cat2000/trainSet/Stimuli')


def fixations_path(datasets_root: str = None) -> str:
    if datasets_root is None:
        datasets_root = datasets_folder
    return os.path.join(datasets_root, 'Cat2000/trainSet/SCANPATHS')


def data_files(datasets_root: str = None) -> NDArray[(Any, 2), np.str]:

    ip = images_path(datasets_root)
    fp = fixations_path(datasets_root)
    print('Images_path: '+ip)
    print('Fixations_path: ' + fp)

    res = []
    for root, dirs, files in os.walk(ip):
        rel_path = os.path.relpath(root, ip)
        for f in files:
            filename,_ = os.path.splitext(f)

            image_file = os.path.join(root, f)
            fixation_file = os.path.join(fp, rel_path, filename+'.mat')

            if not os.path.isfile(image_file) or not os.path.isfile(fixation_file):
                continue

            res.append((image_file,fixation_file))

    res.sort(key=lambda tpl: tpl[0])

    arry = np.asarray(res)
    # noinspection PyTypeChecker
    return arry


def load_data(randomize: bool = True, datasets_root: str = None) -> Iterable[InputData]:
    print('Load data')
    ip = images_path(datasets_root)
    files = data_files(datasets_root)
    print('Num files discoveded in dataset root: %d' % len(files))

    if randomize:
        np.random.shuffle(files)

    for img, fix in files[:,:]:
        name,_ = os.path.splitext(os.path.relpath(img, ip).replace('\\', '/'))

        img_data = Image.open(img)
        fix_data = scipy.io.loadmat(fix)
        fixations_list = fix_data['value']

        image = np.asarray(img_data)
        scanpaths = [swap_columns(np.round(f[0, 0]['data']).astype(np.uint16)) for f in fixations_list[:, 0]]

        yield InputData(
            name=name,
            image_data=image,
            scanpaths=scanpaths
        )

