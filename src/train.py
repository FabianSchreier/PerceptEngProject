# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import argparse
import json
import os
import pickle
import traceback
from datetime import datetime
from typing import Tuple, List, Dict, Any, Type, TypeVar

import keras_compat
import numpy as np
import datasets
import models
from config.train import Config
from data_generator import AugmentedDataFolderGenerator, select_subset, load_images, FolderDataSource, \
    TarArchiveDataSource, DataSource
import callbacks as own_callbacks

import keras.callbacks as callbacks
from keras import Model
from keras.engine.saving import load_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_utils

import tensorflow as tf
from nptyping import NDArray


def fix_history(history):

    if isinstance(history, dict):
        res = {}
        for k, v in history.items():
            pass
        return res

    if isinstance(history, list):
        res = []
        for k, v in history:
            pass
        return res

    if type(history).__module__ == np:
        pass
    return history


def load_sources(config: Config) -> Dict[str, DataSource]:

    if config.datasets:
        return {ds.name: ds for ds in config.datasets}

    sources = {}  # type: Dict[str, DataSource]

    for dataset_path in config.dataset_paths:
        if not os.path.exists(dataset_path):
            print('Could not find dataset %s' % dataset_path)
            exit(-1)

        if os.path.isdir(dataset_path):
            print('Source: Folder %s' % (dataset_path), flush=True)
            source = FolderDataSource(dataset_path)
        elif dataset_path.endswith('.tar') or dataset_path.endswith('.tar.gz'):
            root_folder = os.path.basename(dataset_path)
            if root_folder.endswith('.gz'):
                root_folder = root_folder[:-3]
            if root_folder.endswith('.tar'):
                root_folder = root_folder[:-4]
            print('Source: Tar archive %s with root %s ' % (dataset_path, root_folder), flush=True)
            source = TarArchiveDataSource(dataset_path, root_folder=root_folder)
        else:
            print('Could not open dataset source %s. Unknown type.' % dataset_path)
            exit(-1)
            raise Exception()   # Does not happen, but type-checker complains

        if len(config.dataset_paths) > 1:
            sources[source.name] = source
        else:
            sources[source.name] = source

    return sources


def split_dataset_indices(config: Config, sources: Dict[str, DataSource]) -> Tuple[NDArray[(Any, 4), str], NDArray[(Any, 4), str]]:

    train_set_size_given = config.train_subset_nums or config.train_subset_ratios
    val_set_size_given = config.val_subset_nums or config.val_subset_ratios

    train_set_index = None
    val_set_index = None

    for source in sources.values():
        print('Loading index file for source %s' % source.name, flush=True)
        source_dataset_index = source.load_files_index_with_source_name()
        print('Found %d entries in dataset index files' % source_dataset_index.shape[0], flush=True)

        set_size = source_dataset_index.shape[0]

        if source.name in config.val_subset_nums:
            if config.val_subset_nums[source.name] > -1:
                val_num = config.val_subset_nums[source.name]
            else:
                val_num = set_size * 0.1
        elif source.name in config.val_subset_ratios:
            if config.val_subset_ratios[source.name] > -1:
                val_num = set_size * config.val_subset_ratios[source.name]
            else:
                val_num = set_size * 0.1
        elif not train_set_size_given and not val_set_size_given:  # Neither train nor val set size was specified -> User default 10%
            val_num = set_size * 0.1
        else:   # Either no val set size given (ergo no validation at all) or this set is not used for validation
            val_num = 0

        if source.name in config.train_subset_nums:
            if config.train_subset_nums[source.name] > -1:
                train_num = config.train_subset_nums[source.name]
            else:
                train_num = set_size - val_num
        elif source.name in config.train_subset_ratios:
            if config.train_subset_ratios[source.name] > -1:
                train_num = set_size * config.train_subset_ratios[source.name]
            else:
                train_num = set_size - val_num
        elif not train_set_size_given:  # No train set size was specified, at all -> use what was left over from val set
            train_num = set_size - val_num
        else:   # This set was not in train set size arguments
            train_num = 0

        if train_num + val_num > source_dataset_index.shape[0]:
            raise ValueError('Can\'t split source dataset %s: train subset size %d and evaluation subset size %d is larger than dataset itself (%d entries).' %
                             (source.name, train_num, val_num, set_size))

        both_set_index, _ = select_subset(source_dataset_index, num=(train_num + val_num))
        source_train_subset_index, source_val_subset_index = select_subset(both_set_index, num=train_num)

        train_set_index = source_train_subset_index if train_set_index is None else np.append (train_set_index, source_train_subset_index, axis=0)
        val_set_index   = source_val_subset_index if val_set_index is None else np.append (val_set_index, source_val_subset_index, axis=0)
        print('Selected %d train and %d evaluation entries from %s dataset' % (train_num, val_num, source.name), flush=True)

    return train_set_index, val_set_index


def fetch_model_from_arguments(config: Config) -> Tuple[Model, Model]:

    with tf.device('/cpu:0'):
        loaded_model = False
        if config.pretrained_model_file is not None:
            print('Loading model from %s' % config.pretrained_model_file, flush=True)
            model = load_model(config.pretrained_model_file)
            loaded_model = True

        elif config.model_type == 'baseline':
            print('Generating baseline model', flush=True)
            if config.use_conv_activation:
                model = models.baseline2(input_shape=(*config.input_size, 2))
            else:
                model = models.baseline(input_shape=(*config.input_size, 2))

            if config.use_output_postprocessing:
                models.add_heatmap_layers(model, fixation_sigma=config.postprocessing_fixation_sigma)

        elif config.model_type == 'transfer':
            print('Generating transfer model', flush=True)
            if config.use_conv_activation:
                model = models.transfer2(input_shape=(*config.input_size, 3))
            else:
                model = models.transfer(input_shape=(*config.input_size, 3))

            if config.use_output_postprocessing:
                models.add_heatmap_layers(model, fixation_sigma=config.postprocessing_fixation_sigma)

        else:
            print('Unknown model type %s' % config.model_type, flush=True)
            exit(-1)
        model.summary()


    try:
        trained_model = multi_gpu_utils.multi_gpu_model(model)
        print('Generated multi-gpu model:', flush=True)
        trained_model.summary()

    except Exception as e:
        trained_model = model
        print('Failed to generate multi-gpu model:', flush=True)
        traceback.print_exc()
        print('', flush=True)

    print('Compiling model', flush=True)
    trained_model.compile(
        optimizer=Adam(learning_rate=config.learn_rate),
        loss='mean_squared_error',
    )

    return model, trained_model


def main(config: Config):

    sources = load_sources(config)
    print('Sources: ' + ', '.join(sources.keys()))

    print('output_folder: %s' % config.output_folder, flush=True)
    print('epochs: %d' % config.epochs, flush=True)

    print('input_size: %s' % (config.input_size,), flush=True)
    print('ground_truth_size: %s' % (config.ground_truth_size,), flush=True)

    train_set_index, eval_set_index = split_dataset_indices(config, sources)

    model, trained_model = fetch_model_from_arguments(config)

    print('Loading all data', flush=True)
    start_time = datetime.now()
    X, y = load_images(train_set_index, config.input_size, config.ground_truth_size, sources, third_channel=config.model_third_channel)

    val_X, val_y = load_images(eval_set_index, config.input_size, config.ground_truth_size, sources, third_channel=config.model_third_channel)
    end_time = datetime.now()

    load_time = end_time-start_time
    print('Data loading time: %s' % (load_time))

    start_time = datetime.now()

    cbks = config.get_callbacks()
    cbks.append(own_callbacks.EuclideanDistance())

    history = trained_model.fit(
        x=X, y=y,
        batch_size=75 * 4,
        epochs=config.epochs,
        initial_epoch=config.initial_epoch,
        verbose=1,
        validation_data=(val_X, val_y),
        shuffle=True,
        max_queue_size=20,
        callbacks=cbks
    )  # type: callbacks.History

    end_time = datetime.now()

    train_time = end_time-start_time
    epoch_time = train_time/config.epochs
    print('Training time: %s, per epoch time: %s' % (train_time, epoch_time))

    if config.output_folder is not None:
        print('Exporting model')
        model.save(os.path.join(config.output_folder, 'baseline_model.h5'))

        print('Exporting histories')
        with open(os.path.join(config.output_folder, 'history.pckl'), 'wb') as fp:
            pickle.dump(history.history, fp)

    for source in sources.values():
        source.close()


T = TypeVar('T')


if __name__ == "__main__":
    print('Running train', flush=True)
    config = Config.load_from_commandline()

    main(config)
    exit(0)

