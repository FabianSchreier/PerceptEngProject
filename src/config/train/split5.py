# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

from .split_common import cfg

cfg.train_subset_nums = {
    'cat2000': 0,
    'salicon': 500000,
}
cfg.val_subset_nums = {
    'cat2000': 50000,
    'salicon': 0,
}