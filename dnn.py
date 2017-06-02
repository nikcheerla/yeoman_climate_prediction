# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''

import numpy as np
import warnings, json

import keras
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, UpSampling2D, Cropping2D

from keras.layers import BatchNormalization
from keras.models import Model, load_model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from external.resnet50 import ResNet50
from external.vgg19 import VGG19

import matplotlib.pyplot as plt

import IPython, tempfile, shutil, os







""" Weights utils """

import h5py

def str_shape(x):
    return 'x'.join(map(str, x.shape))










def clear_session_except_model(model):
    model.save("cache.h5")
    K.clear_session()
    model = load_model("cache.h5")
    os.remove("cache.h5")
    return model










""" DNN FCN utils """

def constrain(l1, inputs):
    model = Model(inputs, l1)

    w1 = model.layers[-1].output_shape[2]
    i1 = model.layers[0].batch_input_shape[2]

    if w1 == i1:
        return l1

    s1 = (i1 - w1)//2
    #print ("Inputs", w1, i1)
    w1 = Cropping2D(cropping=((-s1, s1 + w1 - i1), (-s1, s1 + w1 - i1)))(l1)
    return w1


def fully_convolutional(model):
    inputs = model.input_tensor_fcn
    arr = model.tensor_hooks_fcn

    desired_output_shape = model.input_shape

    for i in range(0, len(arr)):
        cur_shape = Model(inputs, arr[i]).output_shape
        scale_factor = desired_output_shape[-2]/cur_shape[-2]
        arr[i] = UpSampling2D((scale_factor, scale_factor)) (arr[i])
        #arr[i] = constrain(arr[i], inputs)

    x = merge(arr, mode='concat', concat_axis=3)
    return Model(inputs, x)

def resize(model, input_shape=(None, None, 3)):
    json = model.get_config()
    json['layers'][0]['config']['batch_input_shape'] = (None, ) + input_shape
    model2 = Model.from_config(json)
    model2.set_weights(model.get_weights())
    return model2

    

def dilation_map(model):

    json = model.get_config()
    
    dilation = {}
    for layer in json['layers']:
        name = layer['name']
        print (name)
        if layer['class_name'] == 'InputLayer':
            dilation[name] = 1
            continue

        prev = layer['inbound_nodes'][0][0][0]
        print (prev)
        dilation[name] = dilation[prev]

        if layer['class_name'] in ['Convolution2D']:
            layer['config']['dilation_rate'] = (dilation[prev], dilation[prev])

        if layer['class_name'] in ['MaxPooling2D', 'Conv2D', 'Convolution2D', 'AveragePooling2D']:
            mul = layer['config']['strides'][0]
            dilation[name] = mul*dilation[prev]
            layer['config']['strides'] = (1, 1)
            layer['config']['padding'] = 'same'

    model2 = Model.from_config(json)
    model2.set_weights(model.get_weights())

    return model2






if __name__ == '__main__':
    model = VGG19(weights=None, input_shape=(64*7, 64*7, 3), classes=2, num_filters=16, pooling='avg')
    model2 = dilation_map(model)

    IPython.embed()

    model2 = fully_convolutional(model)
    model2.summary()

    
    img_path = 'external/Africa.jpg'
    img = image.load_img(img_path, target_size=(64*7, 64*7))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model2.predict(x)

    IPython.embed()
    
    
