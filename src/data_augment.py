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



def get_rescale_factor(source_size: Union[Tuple[int, int], NDArray[2, np.int32]], target_size: Tuple[int, int]) -> float:
    if not isinstance(source_size, np.ndarray):
        source_size = np.asarray(source_size)

    scales = source_size / target_size
    smaller_scale = min(scales)
    return smaller_scale


def get_crop_indices_and_offset(scanpath: NDArray[(Any, 2), np.uint16], size: Tuple[int, int], old_size: Tuple[int, int]) -> Tuple[NDArray[(Any, 1), np.bool], Tuple[int, int]]:
    cropy, cropx = size
    y,x = old_size
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    endx = startx+cropx
    endy = starty+cropy

    offset = (starty, startx)
    idx = np.logical_and(np.logical_and(scanpath[:,0] >= starty, scanpath[:,0] < endy),
                         np.logical_and(scanpath[:,1] >= startx, scanpath[:,1] < endx))

    return idx, offset


def crop_scanpath(scanpath: NDArray[(Any, 2), np.uint16], crop_idx: NDArray[(Any, 1), np.bool], offset: Tuple[int, int]) -> NDArray[(Any, 2), np.uint16]:
    new_sp = scanpath[crop_idx]
    shifted_sp = new_sp - offset
    return shifted_sp


def rescale_scanpath(scanpath: NDArray[(Any, 2), np.uint16], source_size: Tuple[int, int], target_size: Tuple[int, int]) -> NDArray[(Any, 2), np.uint16]:
    # noinspection PyTypeChecker
    resolution = np.asarray(source_size)  # type: NDArray[2, np.int32]
    scale = get_rescale_factor(resolution, target_size)
    scanpaths_scaled = np.round(scanpath / scale).astype(np.uint16)

    return scanpaths_scaled

def crop_image(img: NDArray[(Any, Any, 3), np.uint8], size: Tuple[int, int]) -> NDArray[(Any, Any, 3), np.uint8]:
    cropy, cropx = size
    y,x = (img.shape[0], img.shape[1])
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


def adjust_size(input: InputData, target_input_size: Tuple[int, int], target_ground_truth_size: Tuple[int, int]) -> InputData:

    img = input.image_data
    scanpaths = input.scanpaths
    # noinspection PyTypeChecker
    resolution = np.asarray(img.shape[:2])  # type: NDArray[2, np.int32]

    input_scale = get_rescale_factor(resolution, target_input_size)
    scaled_input_resolution = np.round(resolution / input_scale).astype(np.uint16)

    gt_scale = get_rescale_factor(resolution, target_ground_truth_size)
    scaled_gt_resolution = np.round(resolution / gt_scale).astype(np.uint16)

    gt_to_input_scale = get_rescale_factor(target_ground_truth_size, target_input_size)

    # Always scale to the smaller size to avoid rounding errors
    if gt_to_input_scale > 1:   # GT is larger than input -> scale down to input
        input_scaled_gt_size = np.asarray(target_ground_truth_size) / gt_to_input_scale

        input_crop_size = (int(min(target_input_size[0], input_scaled_gt_size[0])), int(min(target_input_size[1], input_scaled_gt_size[1])))  # Take the smaller dimensions of both
        gt_crop_size = (int(input_crop_size[0] * gt_to_input_scale), int(input_crop_size[1] * gt_to_input_scale))  # And scale it back up to ground truth size

    else:   # GT is smaller than input -> scale down to GT
        gt_scaled_input_size = np.asarray(target_input_size) * gt_to_input_scale

        gt_crop_size = (int(min(target_ground_truth_size[0], gt_scaled_input_size[0])), int(min(target_ground_truth_size[1], gt_scaled_input_size[1])))  # Take the smaller dimensions of both
        input_crop_size = (int(gt_crop_size[0] / gt_to_input_scale), int(gt_crop_size[1] / gt_to_input_scale))  # And scale it back up to input size

    img_scaled = resize(img, scaled_input_resolution, preserve_range=True).astype(np.uint8)
    scanpaths_scaled = [(obs, np.floor(sp / input_scale).astype(np.uint16)) for obs, sp in scanpaths]
    gt_scanpaths_scaled = [(obs, np.floor(sp / gt_scale).astype(np.uint16)) for obs, sp in scanpaths]

    img_adjusted = crop_image(img_scaled, target_input_size)

    scanpaths_adjusted = []
    gt_scanpaths_adjusted = []

    for i, ((in_obs, in_sp), (gt_obs, gt_sp)) in enumerate(zip(scanpaths_scaled, gt_scanpaths_scaled)):
        assert in_obs == gt_obs

        in_idx, in_offset = get_crop_indices_and_offset(in_sp, input_crop_size, scaled_input_resolution)
        gt_idx, gt_offset = get_crop_indices_and_offset(gt_sp, gt_crop_size, scaled_gt_resolution)

        crop_idx = np.logical_and(in_idx, gt_idx)

        cropped_in_sp = crop_scanpath(in_sp, crop_idx, in_offset)
        cropped_gt_sp = crop_scanpath(gt_sp, crop_idx, gt_offset)

        scanpaths_adjusted.append((in_obs, cropped_in_sp))
        gt_scanpaths_adjusted.append((gt_obs, cropped_gt_sp))

    return InputData(
        name=input.name,
        image_data=img_adjusted,
        scanpaths=scanpaths_adjusted,
        gt_scanpaths=gt_scanpaths_adjusted
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


def augment_entry(input: InputData, target_input_size: Tuple[int, int], target_ground_truth_size: Tuple[int, int], merge_radius: float) -> InputData:

    merged_input = InputData(
        name=input.name,
        image_data=input.image_data,
        scanpaths=[],
        gt_scanpaths=[]
    )
    for obs, sp in input.scanpaths:
        merged_sp = merge_close_fixations(sp, merge_radius)
        merged_input.scanpaths.append((obs, merged_sp))

    adjusted_input = adjust_size(merged_input, target_input_size, target_ground_truth_size)
    return adjusted_input

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



