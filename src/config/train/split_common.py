# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import os

from config.train import Config
import config.common as _common

cfg = Config()

cfg.dataset_paths = [
    os.path.join(_common.default_dataset_root(), 'baseline', 'Cat2000'),
    os.path.join(_common.default_dataset_root(), 'baseline', 'SaliCon'),
    os.path.join(_common.default_dataset_root(), 'baseline', 'Mit1003'),
]
cfg.output_folder = _common.default_output_folder()
#cfg.checkpoint_file = _common.default_output_folder()

cfg.model_type = 'baseline'

cfg.input_size = (128, 160)
cfg.ground_truth_size = (32, 48)

cfg.epochs = 150
cfg.learn_rate = 0.005
cfg.learn_rate_halfing = [100]

cfg.early_stop = True
cfg.early_stop_grace_period = 50
cfg.early_stop_restore_best_weights = True

