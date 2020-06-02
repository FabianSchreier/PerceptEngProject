# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

from .split_common import cfg

cfg.train_subset_nums = {
    'cat2000': 500000,
    'salicon': 0,
    'mit1003': 0,
}
cfg.val_subset_nums = {
    'cat2000': 0,
    'salicon': 0,
    'mit1003': 10000,
}