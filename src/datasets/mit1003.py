# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import csv
import os
from typing import Any, Iterable, List, Tuple

import numpy as np
import scipy.io
from PIL import Image
from nptyping import NDArray

from . import datasets_folder, InputData, swap_columns


name = 'Mit1003'


def images_path(datasets_root: str = None) -> str:
    if datasets_root is None:
        datasets_root = datasets_folder
    return os.path.join(datasets_root, 'Mit1003/ALLSTIMULI')


def fixations_path(datasets_root: str = None) -> str:
    if datasets_root is None:
        datasets_root = datasets_folder
    return os.path.join(datasets_root, 'Mit1003/DATA')


def get_observer_names(datasets_root: str = None) -> List[str]:
    fp = fixations_path(datasets_root)

    res = []

    for f in os.listdir(fp):
        full_path = os.path.join(fp, f)
        if not os.path.isdir(full_path):
            continue

        res.append(f)

    return res


def get_input_files(datasets_root: str = None) -> NDArray[(Any, 1), np.str]:
    ip = images_path(datasets_root)

    res = []
    for f in os.listdir(ip):
        image_file = os.path.join(ip, f)
        if not os.path.isfile(image_file):
            continue

        filename,ext = os.path.splitext(f)
        if ext != '.jpg' and ext != '.jpeg':
            continue
        res.append(image_file)

    res.sort(key=lambda tpl: tpl[0])

    arry = np.asarray(res)
    # noinspection PyTypeChecker
    return arry


def get_observer_files(input_file: str, datasets_root: str = None, observer_names: List[str] = None) -> List[Tuple[str, str]]:

    fp = fixations_path(datasets_root)

    if observer_names is None:
        observer_names = get_observer_names(datasets_root)

    file_name, _ = os.path.splitext(os.path.basename(input_file))

    res = []

    for observer in observer_names:

        scanpath_file = os.path.join(fp, observer, file_name+'.csv')

        if not os.path.exists(scanpath_file):
            continue

        res.append((observer, scanpath_file))

    return res


def read_observer_file(file_path: str) -> NDArray[(Any, 2), np.float]:

    res = []
    with open(file_path, 'r') as fh:
        reader = csv.reader(fh, delimiter=',')

        for row in reader:
            res.append((float(row[0]), float(row[1])))

    return np.asarray(res)


def load_data(randomize: bool = True, datasets_root: str = None) -> Iterable[InputData]:
    print('Load data')
    ip = images_path(datasets_root)
    input_files = get_input_files(datasets_root)
    observer_names = get_observer_names(datasets_root)

    if randomize:
        np.random.shuffle(input_files)

    for img in input_files:
        observer_files = get_observer_files(img, datasets_root, observer_names)

        name,_ = os.path.splitext(os.path.relpath(img, ip).replace('\\', '/'))

        img_data = Image.open(img)

        image = np.asarray(img_data)
        scanpaths = []

        for obs, obs_file in observer_files:
            scanpath = read_observer_file(obs_file)
            scanpath = np.round(scanpath).astype(np.uint16)
            scanpath = swap_columns(scanpath)

            scanpaths.append((obs, scanpath))


        yield InputData(
            name=name,
            image_data=image,
            scanpaths=scanpaths,
            gt_scanpaths=[]
        )

