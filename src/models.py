# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

from typing import Optional

import numpy as np
from keras import Model, Sequential, layers, backend
from keras.applications import keras_modules_injection
from keras.applications.resnet import ResNet50
from keras.engine import Layer
import keras.backend as K
from keras.layers import Deconvolution2D, Conv2DTranspose, Conv2D, BatchNormalization, Activation, Flatten, Softmax, \
    DepthwiseConv2D
from keras_applications.resnet_common import ResNet
from scipy.ndimage import gaussian_filter
from tensorflow_core.python import reduce_max, reduce_min


def block0(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='SAME',
                      name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack0(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block0(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block0(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


@keras_modules_injection
def ResNet18(include_top=True,
             weights: Optional[str] ='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    def stack_fn(x):
        x = stack0(x, 64, 2, stride1=2, name='conv2')
        x = stack0(x, 128, 2, name='conv3')
        x = stack0(x, 256, 2, name='conv4')
        x = stack0(x, 512, 2, name='conv5')
        return x
    return ResNet(stack_fn, False, True, 'resnet36',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes,
                  **kwargs)



@keras_modules_injection
def ResNet34(include_top=True,
              weights: Optional[str] ='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs):
    def stack_fn(x):
        x = stack0(x, 64, 3, stride1=2, name='conv2')
        x = stack0(x, 128, 4, name='conv3')
        x = stack0(x, 256, 6, name='conv4')
        x = stack0(x, 512, 3, name='conv5')
        return x
    return ResNet(stack_fn, False, True, 'resnet36',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes,
                  **kwargs)


class Softmax2D(Layer):
    def __init__(self, name: str = None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        x = inputs
        shape = K.shape(inputs)

        x = Flatten()(x)
        x = Softmax()(x)
        x = K.reshape(x, shape)
        return x


class RescaleToByte(Layer):
    def __init__(self, name: str = None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        x = inputs

        max = reduce_max(x, axis=(1, 2), keepdims=True)
        min = reduce_min(x, axis=(1, 2), keepdims=True)

        diff = max - min + 1e-4

        x = ((x - min) / diff) * 256

        return x


def baseline(*, input_shape = None) -> Model:

    if input_shape is None:
        input_shape = (None, None, 2)

    model = Sequential()
    model.add(ResNet34(
        include_top=False,
        weights=None,
        input_shape=input_shape
    ))

    model.add(Conv2DTranspose(
        filters=512,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(Conv2DTranspose(
        filters=256,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(Conv2DTranspose(
        filters=128,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(Conv2DTranspose(
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(Conv2D(
        filters=1,
        kernel_size=3,
        strides=1,
        padding='same',
    ))

    return model


def baseline2(*, input_shape = None) -> Model:

    if input_shape is None:
        input_shape = (None, None, 2)

    model = Sequential()
    model.add(ResNet34(
        include_top=False,
        weights=None,
        input_shape=input_shape
    ))

    model.add(Conv2DTranspose(
        filters=512,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(
        filters=256,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(
        filters=128,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(
        filters=1,
        kernel_size=3,
        strides=1,
        padding='same',
    ))

    return model


def transfer(*, input_shape = None) -> Model:

    if input_shape is None:
        input_shape = (None, None, 3)

    resnet = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,

    )   # type: Model
    resnet.trainable = False

    model = Sequential()
    model.add(resnet)

    model.add(Conv2DTranspose(
        filters=512,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(Conv2DTranspose(
        filters=128,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(Conv2DTranspose(
        filters=32,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(Conv2D(
        filters=1,
        kernel_size=3,
        strides=1,
        padding='same',
    ))

    return model


def transfer2(*, input_shape = None) -> Model:

    if input_shape is None:
        input_shape = (None, None, 3)

    resnet = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,

    )   # type: Model
    resnet.trainable = False

    model = Sequential()
    model.add(resnet)

    model.add(Conv2DTranspose(
        filters=512,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(
        filters=128,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(
        filters=32,
        kernel_size=3,
        strides=2,
        padding='same',
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(
        filters=1,
        kernel_size=3,
        strides=1,
        padding='same',
    ))

    return model


def add_heatmap_layers(input_model: Sequential, fixation_sigma: float):
    kernel_size = int(fixation_sigma*8) + 1

    kernel_weights = np.zeros(shape=(kernel_size, kernel_size))
    kernel_weights[int(fixation_sigma*4), int(fixation_sigma*4)] = 1
    gaussian_filter(kernel_weights, sigma=fixation_sigma, output=kernel_weights)

    kernel_weights.shape = (*kernel_weights.shape, 1, 1)

    #kernel_weights = np.expand_dims(kernel_weights, axis=-1)    # Add one dim for channels (last)
    #kernel_weights = np.expand_dims(kernel_weights, axis=0)     # Add one dim for batch size (first)


    g_layer = DepthwiseConv2D(kernel_size,
                              use_bias=False,
                              padding='same',
                              weights=[kernel_weights],
                              trainable=False,
                              name='heatmap_blur')

    input_model.add(Softmax2D(name='heatmap_softmax'))
    input_model.add(g_layer)
    input_model.add(RescaleToByte(name='heatmap_rescale'))