# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import os
from typing import NamedTuple, Tuple, Any, List

import numpy as np
from nptyping import NDArray

root_folder = os.path.abspath(os.path.join(__file__, '..', '..', '..', '..', '..', '..'))
datasets_folder = os.path.join(root_folder, 'Datasets')
processed_datasets_folder = os.path.join(root_folder, 'ProcessedDatasets')

FixationsType = NDArray[(Any, 2), np.uint16]

InputData = NamedTuple(
    'InputData',
    [
        ('name', str),
        ('image_data', NDArray[(Any, Any, 3), np.uint8]),
        ('scanpaths', List[Tuple[str, NDArray[(Any, 2), np.uint16]]]),
        ('gt_scanpaths', List[Tuple[str, NDArray[(Any, 2), np.uint16]]]),
    ])


def swap_columns(scanpath: NDArray[(Any, 2), np.uint16]) -> NDArray[(Any, 2), np.uint16]:
    scanpath[:, 0], scanpath[:, 1] = scanpath[:, 1], scanpath[:, 0].copy()
    return scanpath  # The above does actually work in-place, but to use this function as right-values we need a return
