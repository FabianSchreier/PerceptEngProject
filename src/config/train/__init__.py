# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import argparse
import importlib
from typing import TYPE_CHECKING, List, Optional, Dict, Type, TypeVar, Callable, Any

import keras.callbacks as callbacks
import callbacks as own_callbacks
from typish import Module

if TYPE_CHECKING:
    from data_generator import DataSource

T = TypeVar('T')


class Config:
    field_aliases = {
        'lr': 'learn_rate',

        'model_file': 'pretrained_model_file',

        'sources': 'datasets',
    }

    def __init__(self):
        self.dataset_paths = []  # type: List[str]
        self.datasets = None     # type: Optional[List[DataSource]]
        self.output_folder = None   # type: Optional[str]

        self.model_type = 'transfer'
        self.tranfer_model_input = 'image'

        self.epochs = 1
        self.initial_epoch = 0
        self.learn_rate = 0.001
        self.learn_rate_halfing = []  # type: List[int]

        self.pretrained_model_file = None  # type: Optional[str]
        self.checkpoint_file = None  # type: Optional[str]
        self.checkpoint_save_best = False

        self.early_stop = False
        self.early_stop_patience = 5
        self.early_stop_grace_period = 0  # type: int
        self.early_stop_restore_best_weights = False

        self.use_output_postprocessing = False
        self.use_conv_activation = False

        self.postprocessing_fixation_sigma = 2.5

        self.input_size = (128, 160)
        self.ground_truth_size = (32, 48)

        self.train_subset_nums = {}  # type: Dict[str, int]
        self.train_subset_ratios = {}  # type: Dict[str, float]
        self.val_subset_nums = {}  # type: Dict[str, int]
        self.val_subset_ratios = {}  # type: Dict[str, float]

    @property
    def model_third_channel(self) -> Optional[str]:
        if self.model_type == 'transfer':
            return self.tranfer_model_input
        return None


    def get_callbacks(self, with_early_stop: bool = True) -> List[callbacks.Callback]:
        res = []

        if self.early_stop:
            print('Adding early stop callback with grace period %d and patience %d' % (self.early_stop_grace_period, self.early_stop_patience), flush=True)
            res.append(own_callbacks.EarlyStoppingWithGracePeriod(
                grace_period=self.early_stop_grace_period,
                verbose=1,
                patience=self.early_stop_patience,
                restore_best_weights=self.early_stop_restore_best_weights
            ))

        if self.checkpoint_file is not None:
            print('Saving checkpoints to %s' % self.checkpoint_file, flush=True)
            res.append(callbacks.ModelCheckpoint(
                filepath=self.checkpoint_file,
                save_best_only=self.checkpoint_save_best
            ))

        if self.learn_rate_halfing:
            print('Adding learn rate halfing at epochs %s' % (', '.join([str(e) for e in self.learn_rate_halfing])), flush=True)

            def lr_half(epoch, lr):
                if epoch in self.learn_rate_halfing:
                    print('Halfing learnrate from %.4f to %.4f at epoch %d' % (lr, lr/2.0, epoch), flush=True)
                    lr = lr / 2.0
                return lr

            res.append(callbacks.LearningRateScheduler(lr_half))

        return res

    @staticmethod
    def load_from_commandline() -> 'Config':
        parser = Config._generate_cmd_parser()
        args = parser.parse_args()

        if args.config:
            print('Loading config from module %s (%s)' % (__name__+'.'+args.config, args.config), flush=True)
            module = importlib.import_module(__name__+'.'+args.config)
            config = Config._try_load_instance(module)
            if config is None:
                config = Config._try_load_fields(module)
        else:
            print('Using default config', flush=True)
            config = Config()

        config._load_commandline_overwrites(args)
        return config

    @staticmethod
    def _try_load_instance(module: Module) -> Optional['Config']:
        inst = getattr(module, 'cfg', None) or getattr(module, 'conf', None) or getattr(module, 'config', None)
        if inst is None or not isinstance(inst, Config):
            return None
        return inst

    @staticmethod
    def _try_load_fields(module: Module) -> 'Config':

        res = Config()
        for f in dir(module):
            if f.startswith('_'):
                continue

            real_f = Config.field_aliases.get(f, f)  # type: str

            value = getattr(module, f)

            if callable(value) or not hasattr(res, real_f):
                print('WARNING: Unused config option %s (un-aliased: %s) in config file %s. No such config field exists. ' % (real_f, f, module.__name__))
                continue

            setattr(res, real_f, value)

        return res

    @staticmethod
    def _generate_cmd_parser() -> argparse.ArgumentParser:

        parser = argparse.ArgumentParser(description='Process dataset')

        parser.add_argument('--config',
                            action='store',
                            help='Root folder for datasets')

        parser.add_argument('--dataset_paths',
                            action='append',
                            help='Root folder for datasets')
        parser.add_argument('--output_folder',
                            action='store',
                            help='Folder for putting output files (model data etc)')

        parser.add_argument('--epochs',
                            type=int,
                            action='store',
                            help='Number of epochs for training')
        parser.add_argument('--initial_epoch',
                            type=int,
                            action='store',
                            help='Start epoch for resumed training')
        parser.add_argument('--lr',
                            type=float,
                            action='store',
                            help='Learning rate for the optimizer')
        parser.add_argument('--learn_rate_halfing',
                            type=int,
                            action='append')

        parser.add_argument('--model_file',
                            type=str,
                            action='store',
                            help='File name of the stored model, if resuming from existing training')
        parser.add_argument('--checkpoint_file',
                            type=str,
                            action='store',
                            help='File name (pattern) for storing checkpoint model files')
        parser.add_argument('--early_stop',
                            type=bool,
                            action='store',
                            help='Stop the trainign process early')
        parser.add_argument('--early_stop_baseline',
                            type=float,
                            action='store',
                            help='Baseline value for early stop callback')

        parser.add_argument('--input_size',
                            type=int,
                            nargs=2,
                            action='store',
                            help='Size of the input images')
        parser.add_argument('--ground_truth_size',
                            type=int,
                            nargs=2,
                            action='store',
                            help='Size of the ouput images')

        parser.add_argument('--train_subset_nums',
                            action='store',
                            help='Number of samples for train set from given dataset. Format: "<dataset name>:<num>[,<dataset2 name>:<num>,...]')
        parser.add_argument('--train_subset_ratios',
                            action='store',
                            help='Ratio of samples for train set from given dataset. Format: "<dataset name>:<ratio>[,<dataset2 name>:<ratio>,...]')
        parser.add_argument('--val_subset_nums',
                            action='store',
                            help='Number of samples for evaluation set from given dataset. Format: "<dataset name>:<num>[,<dataset2 name>:<num>,...]')
        parser.add_argument('--val_subset_ratios',
                            action='store',
                            help='Ratio of samples for evaluation set from given dataset. Format: "<dataset name>:<ratio>[,<dataset2 name>:<ratio>,...]')

        return parser

    def _load_commandline_overwrites(self, args: argparse.Namespace):
        self._load_commandline_overwrite_field(args, 'dataset_paths')
        self._load_commandline_overwrite_field(args, 'output_folder')

        self._load_commandline_overwrite_field(args, 'epochs')
        self._load_commandline_overwrite_field(args, 'initial_epoch')
        self._load_commandline_overwrite_field(args, 'learn_rate', 'lr')
        self._load_commandline_overwrite_field(args, 'learn_rate_halfing')

        self._load_commandline_overwrite_field(args, 'pretrained_model_file', 'model_file')
        self._load_commandline_overwrite_field(args, 'checkpoint_file')
        self._load_commandline_overwrite_field(args, 'early_stop')
        self._load_commandline_overwrite_field(args, 'early_stop_baseline')

        self._load_commandline_overwrite_field(args, 'input_size', process_function=lambda vs: tuple(vs))
        self._load_commandline_overwrite_field(args, 'ground_truth_size', process_function=lambda vs: tuple(vs))

        self._load_commandline_overwrite_field(args, 'train_subset_nums', process_function=lambda v: Config._load_subset_nums(v, int))
        self._load_commandline_overwrite_field(args, 'train_subset_ratios', process_function=lambda v: Config._load_subset_nums(v, float))
        self._load_commandline_overwrite_field(args, 'val_subset_nums', process_function=lambda v: Config._load_subset_nums(v, int))
        self._load_commandline_overwrite_field(args, 'val_subset_ratios', process_function=lambda v: Config._load_subset_nums(v, float))

    def _load_commandline_overwrite_field(self: 'Config', args: argparse.Namespace, field: str, args_name: str = None,
                                          process_function: Callable[[str], Any] = None):
        args_name = args_name or field

        value = getattr(args, args_name, None)
        if value is None:
            return

        if process_function is not None:
            value = process_function(value)

        setattr(self, field, value)

    @staticmethod
    def _load_subset_nums(arg: str, t: Type[T]) -> Optional[Dict[str, T]]:
        if arg is None:
            return None

        res = {}
        for dataset_data in arg.split(','):  # type: str
            name, num = dataset_data.split(':')
            res[name.strip().lower()] = t(num)

        return res

