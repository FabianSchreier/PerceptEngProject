# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import argparse
import json
import os
import pickle

import numpy as np
import datasets
import models
from data_generator import AugmentedDataFolderGenerator, load_files_index, select_subset, load_images
from keras.callbacks import History
from keras.optimizers import Adam
from numpy.random._generator import default_rng#

import tensorflow as tf


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



if __name__ == "__main__":
    print('Running train', flush=True)

    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--dataset',
                        action='store',
                        default=os.path.abspath(os.path.join(datasets.root_folder, '..', '..', '..', 'ProcessedDatasets', 'Cat2000')),
                        help='Root folder for dataset')
    parser.add_argument('--output_folder',
                        action='store',
                        default='.',
                        help='Folder for putting output files (model data etc)')
    parser.add_argument('--workers',
                        type=int,
                        action='store',
                        default=1,
                        help='Number of worker for training')
    parser.add_argument('--epochs',
                        type=int,
                        action='store',
                        default=1000,
                        help='Number of epochs for training')

    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    data_folder = args.dataset
    output_folder = args.output_folder
    print('data_folder: ' + data_folder, flush=True)
    print('output_folder: ' + data_folder, flush=True)
    print('workers: %d' % args.workers, flush=True)

    image_size = (128, 160)
    #image_size = (480, 640)

    print('Loading index file', flush=True)
    whole_dataset_index = load_files_index(data_folder)
    print('Found %d entries in dataset index files' % whole_dataset_index.shape[0], flush=True)

    eval_set_index, train_set_index = select_subset(whole_dataset_index, ratio=0.1)

    '''train_generator = AugmentedDataFolderGenerator(files_index=train_set_index,
                                                   folder=data_folder,
                                                   batch_size=16,
                                                   image_size=image_size,
                                                   shuffle=True)
    val_generator = AugmentedDataFolderGenerator(files_index=eval_set_index,
                                                 folder=data_folder,
                                                 batch_size=16,
                                                 image_size=image_size,
                                                 shuffle=False)
    '''
    print('Loading all data', flush=True)
    X, y = load_images(train_set_index, image_size, data_folder)
    val_X, val_y = load_images(eval_set_index, image_size, data_folder)

    model = models.baseline(input_shape=(*image_size, 2))
    model.summary()

    model.compile(
        optimizer=Adam(),
        loss='mean_squared_error',
        metrics=['accuracy']
    )

    '''history = model.fit_generator(generator=train_generator,
                        validation_data=eval_generator,
                        epochs=args.epochs,
                        verbose=1,
                        validation_freq=1,
                        workers=1)  # type: History
                        '''
    history = model.fit(
        x=X, y=y,
        batch_size=16,
        epochs=args.epochs,
        verbose=1,
        validation_data=(val_X, val_y),
        shuffle=True,
        max_queue_size=20
    )

    print('Exporting model')
    model.save(os.path.join(output_folder, 'baseline_model.h5'))

    print('Exporting history')
    with open(os.path.join(output_folder, 'history.pckl'), 'wb') as fp:
        pickle.dump(history.history, fp)

