# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

from datetime import datetime
from typing import Optional

import numpy as np

import keras.callbacks as callbacks


class EpochTimePrint(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None  # type: Optional[datetime]

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        end_time = datetime.now()

        epoch_time = end_time - self.start_time
        print('#### Actual epoch training time: %s' % (epoch_time))


class EarlyStoppingWithGracePeriod(callbacks.EarlyStopping):
    def __init__(self,
                 grace_period,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):

        self.grace_period = grace_period

        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights
        )

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.grace_period:
            return

        super().on_epoch_end(epoch, logs)


class EuclideanDistance(callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or len(self.validation_data) < 2:
            return

        x_val = self.validation_data[0]
        y_true = self.validation_data[1]
        y_pred = self.model.predict(x_val)

        batch_size = y_true.shape[0]
        distances = np.zeros(shape=batch_size)

        for i in range(y_true.shape[0]):
            max_true = np.asarray(np.unravel_index(np.argmax(y_true[i, :, :, 0]), shape=y_true.shape[1:-1]))
            max_pred = np.asarray(np.unravel_index(np.argmax(y_pred[i, :, :, 0]), shape=y_true.shape[1:-1]))

            dist = np.linalg.norm(max_pred - max_true)
            distances[i] = dist

        dist_avr = np.average(distances)
        dist_var = np.var(distances)
        print('Distance avr: %.3f, var: %.5f' % (dist_avr, dist_var))
        logs['dist_avr'] = dist_avr
        logs['dist_var'] = dist_var
