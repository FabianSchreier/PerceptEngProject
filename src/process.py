# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import argparse
import importlib
import os
import shutil
import time
from concurrent.futures._base import Future
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Tuple

import datasets

import datasets.cat2000 as cat2000
import datasets.salicon as salicon
import datasets.salicon as mit1003
import h5py
import numpy as np
from PIL import Image
from data_augment import adjust_size, to_grayscale, generate_single_fixation_heatmap, generate_scanpath_heatmap, \
    convert_to_bytes, augment_entry, generate_scanpath_heatmap_inplace, generate_single_fixation_heatmap_inplace

'''
def visualize_heatmap(heatmap: NDArray[(Any, Any, 1), np.float], markers: NDArray[(Any, 2), np.uint16] = None) -> Image:

    res = np.zeros(shape=(*heatmap.shape, 3), dtype=np.uint8)

    diff = heatmap.max() - heatmap.min() + 1e-4   # Add small epsilon
    res[...,2] = (((heatmap - heatmap.min()) / diff)*256).astype(np.uint8)    # Normalize to [0->1] and add heatmap as blue channel

    if markers is not None:
        for (y, x) in markers:
            res[y, x, 0] = 255  # Markers on red channel

    img = Image.fromarray(res)
    return img
'''


class Context:
    def __init__(self, *, output_folder: str, images_folder: str, scanpath_folder: str, fixations_folder: str,
                 target_input_size: Tuple[int, int], target_ground_truth_size: Tuple[int, int], fixation_sigma: float):
        self.output_folder = output_folder
        self.images_folder = images_folder
        self.scanpath_folder = scanpath_folder
        self.fixations_folder = fixations_folder

        self.target_input_size = target_input_size
        self.target_ground_truth_size = target_ground_truth_size
        self.fixation_sigma = fixation_sigma


def process_entry(entry: datasets.InputData, ctx: Context):
    print('Processing %s' % entry.name, flush=True)

    entry_folder = os.path.dirname(entry.name)
    entry_filename, _ = os.path.splitext(os.path.basename(entry.name))

    os.makedirs(os.path.join(ctx.images_folder, entry_folder), exist_ok=True)

    entry_augmented = augment_entry(entry, ctx.target_input_size, ctx.target_ground_truth_size, merge_radius=60)
    img_data = to_grayscale(entry_augmented.image_data)

    image_file_name = os.path.join(ctx.images_folder, entry_folder, entry_filename + '.jpg')
    img = Image.fromarray(img_data).convert('L')
    img.save(image_file_name)

    partial_entry_index = []    # type: List[Tuple[str, str, str, str]]

    os.makedirs(os.path.join(ctx.scanpath_folder, entry_folder), exist_ok=True)

    data_index = 0
    for sp_i, ((obs, sp), (gt_obs, gt_sp)) in enumerate(zip(entry_augmented.scanpaths, entry_augmented.gt_scanpaths)):
        assert obs == gt_obs

        for f_i in range(sp.shape[0]-1):

            scanpath_heatmap = convert_to_bytes(generate_scanpath_heatmap(sp, f_i, ctx.target_input_size, ctx.fixation_sigma))
            target_fixation_heatmap = convert_to_bytes(generate_single_fixation_heatmap(gt_sp[f_i+1], ctx.target_ground_truth_size, ctx.fixation_sigma))
            
            os.makedirs(os.path.join(ctx.scanpath_folder, entry_folder, entry_filename), exist_ok=True)
            scanpath_file_name = os.path.join(ctx.scanpath_folder, entry_folder, entry_filename, '%d_%d.jpg' % (sp_i, f_i))
            img = Image.fromarray(scanpath_heatmap).convert('L')
            img.save(scanpath_file_name)

            os.makedirs(os.path.join(ctx.fixations_folder, entry_folder, entry_filename), exist_ok=True)
            fixation_file_name = os.path.join(ctx.fixations_folder, entry_folder, entry_filename, '%d_%d.jpg' % (sp_i, f_i))
            img = Image.fromarray(target_fixation_heatmap).convert('L')
            img.save(fixation_file_name)

            image_file_index = os.path.relpath(image_file_name, ctx.output_folder)
            scanpath_file_index = os.path.relpath(scanpath_file_name, ctx.output_folder)
            fixation_file_index = os.path.relpath(fixation_file_name, ctx.output_folder)

            partial_entry_index.append((image_file_index, scanpath_file_index, fixation_file_index, obs))

    #print('Generated %d entries for  %s' % (len(partial_entry_index), entry.name), flush=True)
    return partial_entry_index



def main():
    print('Running processing')
    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--dataset_root',
                        action='store',
                        help='Root folder for datasets')
    parser.add_argument('--output_root',
                        action='store',
                        help='Root folder for processed datasets')
    parser.add_argument('--parallel_entries',
                        type=int,
                        action='store',
                        default=1,
                        help='Number of worker for training')
    parser.add_argument('--input_size',
                        type=int,
                        nargs=2,
                        action='store',
                        default=(128, 160),
                        help='Size of the input images')
    parser.add_argument('--ground_truth_size',
                        type=int,
                        nargs=2,
                        action='store',
                        default=(32, 48),
                        help='Size of the ouput images')
    parser.add_argument('--dataset',
                        action='store',
                        default='Cat2000',
                        help='Name of teh dataset module')

    args = parser.parse_args()

    dataset_root = args.dataset_root

    output_root = args.output_root
    if output_root is None:
        output_root = os.path.join(datasets.root_folder, 'ProcessedDatasets')

    print('dataset_root: %s ' % (os.path.abspath(dataset_root) if dataset_root else 'None'))
    print('output_root: %s ' % os.path.abspath(output_root))
    print('parallel_entries: %d' % args.parallel_entries)
    print('dataset: %s' % args.dataset.lower())

    try:
        dataset = importlib.import_module(datasets.__name__ + '.' + args.dataset.lower())
    except ImportError:
        print('Could not find dataset %s (%s)' % (args.dataset, datasets.__name__ + '.' + args.dataset.lower()))
        exit(-1)

    target_input_size = tuple(args.input_size)
    target_ground_truth_size = tuple(args.ground_truth_size)

    fixation_sigma = 2.5

    one_file_per_entry = True

    output_folder = os.path.join(output_root, dataset.name)
    images_folder = os.path.join(output_folder, 'Images', 'Grayscale')

    #scanpath_folder = os.path.join(output_folder, 'Scanpaths')

    scanpath_folder = os.path.join(output_folder, 'ScanpathHeatmaps')
    fixations_folder = os.path.join(output_folder, 'FixationHeatmaps')
    #scanpath_viz_folder = os.path.join(output_folder, 'ScanpathsViz')

    print('output_folder: '+output_folder)
    print('images_folder: '+images_folder)
    print('scanpath_folder: '+scanpath_folder)

    print('Creating output folder')
    os.makedirs(output_folder, exist_ok=True)

    print('Creating images folder')
    if os.path.exists(images_folder):
        shutil.rmtree(images_folder)
    os.makedirs(images_folder, exist_ok=True)

    print('Creating scanpaths folder')
    if os.path.exists(scanpath_folder):
        shutil.rmtree(scanpath_folder)
    os.makedirs(scanpath_folder, exist_ok=True)

    print('Creating fixations folder')
    if os.path.exists(fixations_folder):
        shutil.rmtree(fixations_folder)
    os.makedirs(fixations_folder, exist_ok=True)

    #if os.path.exists(scanpath_viz_folder):
    #    shutil.rmtree(scanpath_viz_folder)
    #os.makedirs(scanpath_viz_folder, exist_ok=True)

    entry_index = []  # type: List[Tuple[str, str, str, str, str]]

    print('Loading dataset', flush=True)
    data = dataset.load_data(randomize=False, datasets_root=dataset_root)

    ctx = Context(
        fixation_sigma=fixation_sigma,
        target_input_size=target_input_size,
        target_ground_truth_size=target_ground_truth_size,

        output_folder=output_folder,
        images_folder=images_folder,
        fixations_folder=fixations_folder,
        scanpath_folder=scanpath_folder
    )

    def count_futures(entry_index_futures) -> Tuple[int, int]:
        completed = 0
        for f in entry_index_futures:
            if f.done():
                completed += 1

        return completed, len(entry_index_futures)-completed

    '''
    for entry in data:
        if entry.name != 'Action/097':
            continue
        res = process_entry(entry, ctx)
        break
    '''

    with ThreadPoolExecutor() as executor:
        entry_index_futures = []        # type: List[Future[List[Tuple[str, str, str, str]]]]
        completed, not_completed = count_futures(entry_index_futures)

        for entry in data:
            #entry_index.append(process_entry(entry, ctx))


            partial_index_future = executor.submit(process_entry, entry, ctx)
            entry_index_futures.append(partial_index_future)

            completed, not_completed = count_futures(entry_index_futures)

            if completed > 0:
                for future in entry_index_futures:
                    if not future.done():
                        continue
                    partial = future.result()
                    entry_index.extend(partial)
                entry_index_futures = [f for f in entry_index_futures if not f.done()]

            while not_completed >= args.parallel_entries:
                time.sleep(1)
                completed, not_completed = count_futures(entry_index_futures)

    for future in entry_index_futures:
        partial = future.result()
        entry_index.extend(partial)

    index_arry = np.asarray(entry_index)
    np.save(os.path.join(output_folder, 'index.npy'), index_arry)


if __name__ == '__main__':
    main()