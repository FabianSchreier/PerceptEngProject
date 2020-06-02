# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"


import tensorflow as tf
import keras.backend.tensorflow_backend as tfback


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    # global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        if tfback._is_tf_1():
            devices = tfback.get_session().list_devices()
            _LOCAL_DEVICES = [x.name for x in devices]
        else:
            devices = tf.config.list_logical_devices()
            tfback._LOCAL_DEVICES = [x.name for x in devices]
    gpus = [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
    print("GPUs:", ', '.join(gpus))
    return gpus


def fix_multi_gpu_model():

    tfback._get_available_gpus = _get_available_gpus

