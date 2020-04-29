# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

from typing import List, Any, Tuple, Union, cast
import numpy as np
import numpy.linalg as np_la
from skimage.transform import resize
from skimage.util import crop
from scipy.ndimage import gaussian_filter

from datasets import InputData
from nptyping import NDArray


def crop_image(img: NDArray[(Any, Any, 3), np.uint8], size: Tuple[int, int]) -> NDArray[(Any, Any, 3), np.uint8]:
    cropy, cropx = size
    y,x = (img.shape[0], img.shape[1])
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


def crop_scanpaths(scanpaths: List[NDArray[(Any, 2), np.uint16]], size: Tuple[int, int], old_size: Tuple[int, int]) -> List[NDArray[(Any, 2), np.uint16]]:
    cropy, cropx = size
    y,x = old_size
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    endx = startx+cropx
    endy = starty+cropy

    offset = (starty, startx)

    res = []
    for sp in scanpaths:
        new_sp = sp
        new_sp = new_sp[np.logical_and(new_sp[:,0] >= starty, new_sp[:,0] < endy)]
        new_sp = new_sp[np.logical_and(new_sp[:,1] >= startx, new_sp[:,1] < endx)]

        shifted_sp = new_sp - offset
        res.append(shifted_sp)

    return res


def get_rescale_factor(source_size: Union[Tuple[int, int], NDArray[2, np.int32]], target_size: Tuple[int, int]) -> float:
    if not isinstance(source_size, np.ndarray):
        source_size = np.asarray(source_size)

    scales = source_size / target_size
    smaller_scale = min(scales)
    return smaller_scale


def adjust_size(input: InputData, size: Tuple[int, int], rescale: bool = True) -> InputData:

    img = input.image_data
    scanpaths = input.scanpaths
    # noinspection PyTypeChecker
    resolution = np.asarray(img.shape[:2])  # type: NDArray[2, np.int32]

    if rescale:
        scale = get_rescale_factor(resolution, size)
        target_resolution = np.round(resolution / scale).astype(np.uint16)

        img_scaled = resize(img, target_resolution, preserve_range=True).astype(np.uint8)
        scanpaths_scaled = [np.round(sp / scale).astype(np.uint16) for sp in scanpaths]
    else:
        img_scaled = img
        scanpaths_scaled = scanpaths

    img_adjusted = crop_image(img_scaled, size)
    scanpaths_adjusted = crop_scanpaths(scanpaths_scaled, size, img_scaled.shape[:2])

    return InputData(
        name=input.name,
        image_data=img_adjusted,
        scanpaths=scanpaths_adjusted
    )


def merge_close_fixations(scanpath: NDArray[(Any, 2), np.uint16], radius: float) -> NDArray[(Any, 2), np.uint16]:

    remove = np.ones(shape=scanpath.shape[0], dtype=np.bool)
    for i, f in enumerate(scanpath[..., :]):
        for ni, nf in enumerate(scanpath[i+1:, :]):
            d = np_la.norm(nf - f)
            if d > radius:
                break
            remove[i+ni+1] = False

    merged_scanpath = scanpath[remove]
    return merged_scanpath


def to_grayscale(image: NDArray[(Any, Any, 3), np.uint8]) -> NDArray[(Any, Any, 1), np.uint8]:
    if len(image.shape) == 2 or image.shape[2] == 1:
        return image

    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def generate_single_fixation_heatmap(fixation: NDArray[2, np.uint16], size: Tuple[int, int], sigma: float) -> NDArray[(Any, Any), np.float]:
    # noinspection PyTypeChecker
    res = np.zeros(shape=size, dtype=np.float)  # type: NDArray[(Any, Any), np.float]

    generate_single_fixation_heatmap_inplace(res, fixation, sigma)

    return res


def generate_single_fixation_heatmap_inplace(target_array: NDArray[(Any, Any, 1), np.float], fixation: NDArray[2, np.uint16], sigma: float):
    f_y, f_x = fixation

    target_array[f_y, f_x] = 1
    gaussian_filter(target_array, sigma, output=target_array)


def generate_scanpath_heatmap(scanpaths: NDArray[(Any, 2), np.uint16], fixation_index: int, size: Tuple[int, int], sigma: float, fadeoff_factor: float = 2) -> NDArray[(Any, Any, 1), np.float]:
    # noinspection PyTypeChecker
    res = np.zeros(shape=size, dtype=np.float)  # type: NDArray[(Any, Any), np.float]

    generate_scanpath_heatmap_inplace(res, scanpaths, fixation_index, sigma, fadeoff_factor)

    return res


def generate_scanpath_heatmap_inplace(target_array: NDArray[(Any, Any, 1), np.float], scanpaths: NDArray[(Any, 2), np.uint16], fixation_index: int, sigma: float, fadeoff_factor: float = 2):

    selected_fixations = scanpaths[fixation_index::-1, :]
    for i, (f_y, f_x) in enumerate(selected_fixations):
        value = 1 / pow(fadeoff_factor, i)
        if target_array[f_y, f_x] == 0:  # Don't overwrite existing fixations
            target_array[f_y, f_x] = value

    gaussian_filter(target_array, sigma, output=target_array)


def convert_to_bytes(heatmap: NDArray[(Any, Any), np.float]) -> NDArray[(Any, Any), np.uint8]:
    diff = heatmap.max() - heatmap.min() + 1e-4   # Add small epsilon
    res = (((heatmap - heatmap.min()) / diff)*256).astype(np.uint8)    # Normalize to [0->1]
    return res


def generate_input_tensor_grayscale(image: NDArray[(Any, Any, 3), np.uint8], scanpath: NDArray[(Any, 2), np.uint16], fixation_index: int, sigma: float) -> NDArray[(Any, Any, 2), np.float]:
    size = image.shape[:2]

    res = np.zeros(shape=(*size, 2))

    res[...,0] = to_grayscale(image).copy()
    generate_scanpath_heatmap_inplace(res[..., 1], scanpath, fixation_index, sigma)

    return res



