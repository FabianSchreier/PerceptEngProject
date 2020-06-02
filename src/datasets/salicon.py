# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"



import os
from typing import List, Tuple, Any, Iterable

import mat4py
import numpy as np
import scipy.io
from PIL import Image
from nptyping import NDArray

from . import datasets_folder, InputData, swap_columns

name = 'SaliCon'

def images_path(datasets_root: str = None) -> str:
    if datasets_root is None:
        datasets_root = datasets_folder
    return os.path.join(datasets_root, 'SaliCon/images/train')


def fixations_path(datasets_root: str = None) -> str:
    if datasets_root is None:
        datasets_root = datasets_folder
    return os.path.join(datasets_root, 'SaliCon/fixations/train')


def data_files(datasets_root: str = None) -> NDArray[(Any, 2), np.str]:

    ip = images_path(datasets_root)
    fp = fixations_path(datasets_root)

    res = []

    for f in os.listdir(ip):
        if not f.endswith('.jpg'):
            continue
        filename,_ = os.path.splitext(f)
        image_file = os.path.join(ip, f)
        fixation_file = os.path.join(fp, filename+'.mat')
        if not os.path.isfile(image_file) or not os.path.isfile(fixation_file):
            continue

        res.append((image_file,fixation_file))

    arry = np.asarray(res)
    # noinspection PyTypeChecker
    return arry


def load_data(randomize: bool = True, datasets_root: str = None) -> Iterable[InputData]:
    files = data_files(datasets_root)

    if randomize:
        np.random.shuffle(files)

    for img, fix in files[:,:]:
        img_data = Image.open(img)
        fix_data = scipy.io.loadmat(fix)

        name = fix_data['image'][0]
        image = np.asarray(img_data)
        scanpaths = [(str(obs_i), swap_columns(sp)) for obs_i, sp in enumerate(np.squeeze(fix_data['gaze']['fixations']))]

        yield InputData(
            name=name,
            image_data=image,
            scanpaths=scanpaths,
            gt_scanpaths=[]
        )




