import os
import sys
import time
import numpy as np
import tensorflow as tf
import math
from random import getrandbits

from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Activation, Input
from keras.layers import Concatenate
from keras.layers import Add, Multiply, Conv2D, Lambda, Reshape, ZeroPadding2D

from copy import deepcopy

def Concat(x):
    return tf.concat(x, axis=-1)

def L2norm(x):
    return x / K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))

def MOD(x,a):
    return tf.math.mod(x,a)

def EQ(x,a):
    return tf.cast(tf.equal(x,a), dtype='float32')

def FL(x):
    return tf.floor(x)

def MulConst(x,y):
    return x * y

def KAtan2(x):
    return tf.atan2(x[1], x[0])

def sameshape(input_shape):
    return input_shape

def KAtan2_shape(input_shape):
    return input_shape[0]

def CircularGaussKernel(kernlen=21):
    halfSize = kernlen / 2;
    r2 = halfSize*halfSize;
    sigma2 = 0.9 * r2;
    disq = 0;
    kernel = np.zeros((kernlen,kernlen))
    for y in range(kernlen):
        for x in range(kernlen):
            disq = (y - halfSize)*(y - halfSize) +  (x - halfSize)*(x - halfSize);
            if disq < r2:
                kernel[y,x] = math.exp(-disq / sigma2)
            else:
                kernel[y,x] = 0
    return kernel

def get_bin_weight_kernel_size_and_stride(patch_size, num_spatial_bins):
    bin_weight_stride = int(round(2.0 * math.floor(patch_size / 2) / float(num_spatial_bins + 1)))
    bin_weight_kernel_size = int(2 * bin_weight_stride - 1);
    return bin_weight_kernel_size, bin_weight_stride

def get_sift_model(feed, img_rows = 65, num_ang_bins = 4, num_spatial_bins = 4, clipval = 0.2):
    gk = CircularGaussKernel(kernlen=img_rows)
    gauss_kernel = K.variable(value=gk)
    grad_x = Conv2D(1, (3, 1), name = 'gx_' + str(getrandbits(20)))(feed)
    grad_x = ZeroPadding2D(padding=(1, 0))(grad_x)
    grad_x = Reshape((img_rows, img_rows))(grad_x)
    grad_y = Conv2D(1, (1, 3), name = 'gy_' + str(getrandbits(20)))(feed)
    grad_y = ZeroPadding2D(padding=(0,1))(grad_y)
    grad_y = Reshape((img_rows, img_rows))(grad_y)
    grad_x_2 = Lambda(lambda x: x ** 2)(grad_x)
    grad_y_2 = Lambda(lambda x: x ** 2)(grad_y)
    grad_sum_sq = Add()([grad_x_2, grad_y_2])
    magnitude = Lambda(lambda x: x ** 0.5)(grad_sum_sq)
    gauss_weighted_magn = Lambda(MulConst, arguments={'y': gauss_kernel})(magnitude)
    angle_shape = KAtan2_shape(grad_x)
    angle = Lambda(lambda x: KAtan2(x), output_shape = angle_shape)((grad_x, grad_y))
    o_big = Lambda(lambda x: (x + 2.0*math.pi)/ (2.0*math.pi) * float(num_ang_bins))(angle)
    bo0_big = Lambda(FL)(o_big)
    munis_bo0_big = Lambda(lambda x: -x)(bo0_big )
    wo1_big = Add()([o_big, munis_bo0_big])
    bo0_big = Lambda(MOD, arguments = {'a':num_ang_bins})(bo0_big)
    bo0_big_plus1 = Lambda(lambda x: (x  +1.))(bo0_big) 
    bo1_big = Lambda(MOD, arguments = {'a':num_ang_bins})(bo0_big_plus1)    
    wo0_big = Lambda(lambda x: 1. - x)(wo1_big)
    wo0_big = Multiply()([wo0_big, gauss_weighted_magn])
    wo1_big = Multiply()([wo1_big, gauss_weighted_magn])

    ang_bins = []
    bin_weight_kernel_size, bin_weight_stride = get_bin_weight_kernel_size_and_stride(img_rows, num_spatial_bins)
    for i in range(0, num_ang_bins):
        mask1 =  Lambda(EQ, arguments = {'a': i})(bo0_big)
        amask1 =  Lambda(EQ, arguments = {'a': i})(bo1_big)
        weights1 = Multiply()([mask1,wo0_big])
        weights11 = Multiply()([amask1,wo1_big])
        ori0 =  Add()([weights1, weights11])
        ori0 = Reshape((img_rows, img_rows, 1))(ori0)
        bin_weight = Conv2D(1, (bin_weight_kernel_size, bin_weight_kernel_size), 
                                   strides = [bin_weight_stride, bin_weight_stride], 
                                   name = 'bin_weight_' + str(getrandbits(20)))(ori0)
        bin_weight = Flatten()(bin_weight)
        ang_bins.append(bin_weight)
    
    ang_bin_merged = Concatenate()(ang_bins)
    flatten = ang_bin_merged
    l2norm =  Lambda(L2norm)(flatten)
    clipping =  Lambda(lambda x: K.minimum(x,clipval))(l2norm)
    l2norm_again = Lambda(L2norm)(clipping)
    # l2norm_again = Reshape((64, 1))(l2norm_again)
    return l2norm_again

def getPoolingKernel(kernel_size = 25):
    step = 1. / (1e-5 + float(kernel_size // 2))
    x_coef = np.arange(step/2., 1. ,step)
    xc2 = np.hstack([x_coef,[1], x_coef[::-1]])
    kernel = np.outer(xc2.T,xc2)
    kernel = np.maximum(0,kernel)
    return kernel

def initializeSIFT(model):
    for layer in model.layers:
        l_name = layer.get_config()['name']
        w_all = layer.get_weights()
        if l_name[0:2] == 'gy':
            new_weights = np.array([-1, 0, 1], dtype=np.float32)
            new_weights = np.reshape(new_weights, w_all[0].shape)
        elif l_name[0:2] == 'gx':
            new_weights = np.array([-1, 0, 1], dtype=np.float32)
            new_weights = np.reshape(new_weights, w_all[0].shape)
        elif 'bin_weight' in l_name:
            kernel_size  = w_all[0].shape[0]
            nw = getPoolingKernel(kernel_size=kernel_size)
            new_weights = np.array(nw.reshape((kernel_size, kernel_size, 1, 1)))
        else:
            continue
        biases = np.array(w_all[1])
        biases[:] = 0
        w_all_new = [new_weights, biases]
        layer.set_weights(w_all_new)
    return model

''' Can be used as the first layer of a larger model '''
def getSIFTModel(inputs=None, patch_size = 65, num_ang_bins = 4, num_spatial_bins = 4):
    if inputs is None:
        inputs = tf.keras.layers.Input(shape=(patch_size, patch_size, 1))
    # assert shape is n, n, 1
    kerassift = get_sift_model(inputs, img_rows=patch_size, num_ang_bins=num_ang_bins, num_spatial_bins=num_spatial_bins)
    model = Model(inputs=inputs, outputs=kerassift)
    model = initializeSIFT(model)
    model.trainable = False
    return model

def getCompiledSIFTModel(patch_size = 65):
    inputs = Input((patch_size, patch_size, 1), name='main_input')
    kerassift = get_sift_model(inputs)
    model = Model(inputs=inputs, outputs=kerassift)
    model.compile(optimizer='Adam', loss='mse')
    model = initializeSIFT(model)
    model.trainable = False
    return model
