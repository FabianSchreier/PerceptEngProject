# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import numpy as np
from keras import Sequential
from keras.engine import Layer
from keras.layers import Softmax, DepthwiseConv2D, Flatten, Lambda
from models import add_heatmap_layers
from scipy.ndimage import gaussian_filter
from tensorflow_core.python import reduce_max, reduce_min

fixation_sigma = 2.5


# the rest of the model definition...

# do this BEFORE calling `compile` method of the model
#g_layer.set_weights([kernel_weights])



model = Sequential()


add_heatmap_layers(model)



#model.compile(optimizer=None)

input = np.zeros(shape=(3, 21, 21, 1))

input[0, 3, 3, 0] = 1827

input[1, 9, 7, 0] = 10

input[2, 15, 13, 0] = 100
input[2, 3, 2, 0] = 100


output = model.predict(input)

#output[0, :, :, 0]

print()