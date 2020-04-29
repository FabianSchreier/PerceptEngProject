# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

from typing import Optional

from keras import Model, Sequential, layers, backend
from keras.applications import keras_modules_injection
from keras.applications.resnet import ResNet50
from keras.layers import Deconvolution2D, Conv2DTranspose
from keras_applications.resnet_common import stack1, ResNet


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
        x = stack0(x, 64, 2, stride1=1, name='conv2')
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
        x = stack0(x, 64, 3, stride1=1, name='conv2')
        x = stack0(x, 128, 4, name='conv3')
        x = stack0(x, 256, 6, name='conv4')
        x = stack0(x, 512, 3, name='conv5')
        return x
    return ResNet(stack_fn, False, True, 'resnet36',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes,
                  **kwargs)


def baseline(*, grayscale: bool = True, input_shape = None) -> Model:

    if input_shape is None:
        input_shape = (None, None, 2) if grayscale else (None, None, 4)

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
    model.add(Conv2DTranspose(
        filters=1,
        kernel_size=3,
        strides=2,
        padding='same',
    ))

    return model



