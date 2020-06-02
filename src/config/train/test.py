# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"


from .split_common import cfg

cfg.train_subset_nums = {
    'cat2000': 1000,
}
cfg.val_subset_nums = {
    'cat2000': 100,
}

cfg.learn_rate_halfing = []
cfg.epochs = 1
cfg.output_folder = None