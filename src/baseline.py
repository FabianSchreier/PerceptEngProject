# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import os
from typing import Iterable, List, Tuple, Any

import datasets
import numpy as np
from config.train import Config
from data_augment import augment_entry
from datasets import InputData, cat2000, mit1003, salicon
from nptyping import NDArray
from train import load_sources


class Context:
    def __init__(self, *,
                 target_input_size: Tuple[int, int], target_ground_truth_size: Tuple[int, int], fixation_sigma: float):
        self.target_input_size = target_input_size
        self.target_ground_truth_size = target_ground_truth_size
        self.fixation_sigma = fixation_sigma


def get_fixations(dataset, ctx: Context) -> NDArray[Any, 2]:
    fixations_file = 'fixations_%s.npy' % dataset.name
    if os.path.exists(fixations_file):
        return np.load(fixations_file)

    entries = dataset.load_data()  # type: Iterable[InputData]
    fixations = []
    for entry in entries:
        entry_augmented = augment_entry(entry, ctx.target_input_size, ctx.target_ground_truth_size, merge_radius=60)

        for _, sp in entry_augmented.gt_scanpaths:
            for fix in sp:
                fixations.append(fix)
                if (len(fixations) % 10000) == 0:
                    print(len(fixations))

    fixations = np.asarray(fixations)
    np.save(fixations_file, fixations)
    return fixations


def get_avr_dist(fixations: NDArray[Any, 2], ctx: Context):

    avr_fix = np.average(fixations, axis=0)

    center = np.asarray(ctx.target_ground_truth_size) / 2

    dist_to_center = np.linalg.norm(fixations - center, axis=1)
    dist_avr_pos = np.linalg.norm(fixations - avr_fix, axis=1)

    avr_dist_to_center = np.average(dist_to_center)
    std_dist_to_center = np.std(dist_avr_pos)
    avr_dist_avr_pos = np.average(dist_avr_pos)
    std_dist_avr_pos = np.std(dist_avr_pos)
    print()


def main():

    config = Config()
    config.dataset_paths = [
        os.path.abspath(os.path.join(datasets.processed_datasets_folder, 'baseline', 'Cat2000')),
        os.path.abspath(os.path.join(datasets.processed_datasets_folder, 'baseline', 'SaliCon')),
        os.path.abspath(os.path.join(datasets.processed_datasets_folder, 'baseline', 'Mit1003')),
    ]



    ctx = Context(
        fixation_sigma=2.5,
        target_input_size=(128, 160),
        target_ground_truth_size=(32, 48),
    )

    sources = load_sources(config)

    mit_fix = get_fixations(salicon, ctx)
    get_avr_dist(mit_fix, ctx)

    print()





if __name__ == '__main__':
    main()