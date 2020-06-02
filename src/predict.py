# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import os
import shutil

import numpy as np

import datasets
import PIL.Image as Image
from data_generator import load_files_index, load_images
from keras import Model
from keras.engine.saving import load_model


def main():

    input_size = (128, 160)
    ground_truth_size = (32, 48)

    dataset_folder = os.path.abspath(os.path.join(datasets.root_folder, '..', '..', '..', 'ProcessedDatasets', 'Cat2000'))

    prediction_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'predictions'))

    print('Loading index file', flush=True)
    whole_dataset_index = load_files_index(dataset_folder)

    row_i = np.random.choice(whole_dataset_index.shape[0], size=50)
    samples = whole_dataset_index[row_i, :]

    print('Creating predictions folder')
    if os.path.exists(prediction_folder):
        shutil.rmtree(prediction_folder)
    os.makedirs(prediction_folder, exist_ok=True)

    print('Loading model')
    model = load_model('../models/baseline_model.split5-260437.h5')     # type: Model

    print('Loading samples')
    X, y = load_images(samples, input_size, ground_truth_size, dataset_folder)

    print('Predicting samples')
    y_ = model.predict(X)

    print('Saving predictions')
    for i in range(samples.shape[0]):

        img = Image.fromarray(X[i, :, :, 0]).convert('L')
        img.save(os.path.join(prediction_folder, '%d.a.in1.jpg' % i))

        img = Image.fromarray(X[i, :, :, 1]).convert('L')
        img.save(os.path.join(prediction_folder, '%d.b.in2.jpg' % i))

        img = Image.fromarray(y[i, :, :, 0]).convert('L')
        img.save(os.path.join(prediction_folder, '%d.c.gt.jpg' % i))

        img = Image.fromarray(y_[i, :, :, 0]).convert('L')
        img.save(os.path.join(prediction_folder, '%d.d.pr.jpg' % i))

    print('Finished')


if __name__ == '__main__':
    main()

