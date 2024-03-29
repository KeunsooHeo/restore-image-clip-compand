#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.activation import ramp
from tensorlayer.prepro import imresize
import numpy as np
import time
from config import config
# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py

def unet(t_image, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02) #
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    nx = int(t_image._shape[1])
    ny = int(t_image._shape[2])
    nz = int(t_image._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        inputs = InputLayer(t_image, name='inputs')
        conv1_1 = Conv2d(inputs, 16, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv1_1')
        print('conv1_1 : ', conv1_1.outputs)

        ##  conv2
        rconv1_1 = conv1_1
        rconv1_1.outputs = tf.nn.relu(rconv1_1.outputs)
        conv2_1 = Conv2d(rconv1_1, 32, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv2_1')
        print('conv2_1 : ', conv2_1.outputs)
        rconv2_1 = conv2_1
        rconv2_1.outputs = tf.nn.relu(rconv2_1.outputs)
        conv2_2 = Conv2d(rconv2_1, 32, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv2_2')
        print('conv2_2 : ', conv2_2.outputs)
        conv2_3 = conv2_2
        # conv2_3 = separate_fn(conv2_2,conv1_1, 2)
        # print('conv2_3 : ', conv2_3.outputs)
        pool2 = MaxPool2d(conv2_3, (2, 2), padding='SAME', name='pool2')
        print('pool2 : ', pool2.outputs)

        ##  conv3
        rpool2 = pool2
        rpool2.outputs = tf.nn.relu(rpool2.outputs)
        conv3_1 = Conv2d(rpool2, 64, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv3_1')
        print('conv3_1 : ', conv3_1.outputs)
        rconv3_1 = conv3_1
        rconv3_1.outputs = tf.nn.relu(rconv3_1.outputs)
        conv3_2 = Conv2d(rconv3_1, 64, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv3_2')
        print('conv3_2 : ', conv3_2.outputs)
        conv3_3 = conv3_2
        # conv3_3 = separate_fn(conv3_2, pool2, 3)
        print('conv3_3 : ', conv3_3.outputs)
        pool3 = MaxPool2d(conv3_3, (2, 2), padding='SAME', name='pool3')
        print('pool3 : ', pool3.outputs)

        ##  conv4
        rpool3 = pool3
        rpool3.outputs = tf.nn.relu(rpool3.outputs)
        conv4_1 = Conv2d(rpool3, 128, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv4_1')
        print('conv4_1 : ', conv4_1.outputs)
        rconv4_1 = conv4_1
        rconv4_1.outputs = tf.nn.relu(rconv4_1.outputs)
        conv4_2 = Conv2d(rconv4_1, 128, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv4_2')
        print('conv4_2 : ', conv4_2.outputs)
        conv4_3 = conv4_2
        # conv4_3 = separate_fn(conv4_2, pool3, 4)
        print('conv4_3 : ', conv4_3.outputs)
        pool4 = MaxPool2d(conv4_3, (2, 2), padding='SAME', name='pool4')
        print('pool4 : ', pool4.outputs)

        ##  conv5
        rpool4 = pool4
        rpool4.outputs = tf.nn.relu(rpool4.outputs)
        conv5_1 = Conv2d(rpool4, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv5_1')
        print('conv5_1 : ', conv5_1.outputs)
        rconv5_1 = conv5_1
        rconv5_1.outputs = tf.nn.relu(rconv5_1.outputs)
        conv5_2 = Conv2d(rconv5_1, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv5_2')
        print('conv5_2 : ', conv5_2.outputs)
        conv5_3 = conv5_2
        # conv5_3 = separate_fn(conv5_2, pool4, 5)
        print(" * After conv: %s" % conv5_2.outputs)
        pool5 = MaxPool2d(conv5_3, (2, 2), padding='SAME', name='pool5')
        print('pool5 : ', pool5.outputs)

        ##  Bridge
        rpool5 = pool5
        rpool5.outputs = tf.nn.relu(rpool5.outputs)
        Bridge1 = Conv2d(rpool5, 512, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='Bridge1')
        print('conv5_1 : ', conv5_1.outputs)
        rBridge1 = Bridge1
        rBridge1.outputs = tf.nn.relu(rBridge1.outputs)
        Bridge2 = Conv2d(rBridge1, 512, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='Bridge2')
        print('conv5_2 : ', conv5_2.outputs)
        Bridge3 = Bridge2
        # Bridge3 = upsep2_fn(Bridge2, pool5, 6)
        print(" * After conv: %s" % Bridge3.outputs)

        ##  up5
        rBridge3 = Bridge3
        rBridge3.outputs = tf.nn.relu(rBridge3.outputs)
        # up5_0 = UpSampling2dLayer(rBridge3, size=(2, 2), is_scale=True, method=1, name='up5_0')
        # up5_1 = Conv2d(up5_0, 256, (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
        #                  name='up5_1')
        up5_1 = DeConv2d(rBridge3, 256, (2, 2), strides=(2, 2), padding='SAME', W_init=w_init, b_init=b_init, name='up5_1')
        # print('up5 : ', up5_1.outputs)
        up5_2 = ConcatLayer([up5_1, conv5_3], concat_dim=3, name='up5_2')
        print('concat5 : ', up5_2.outputs)
        rup5_2 = up5_2
        rup5_2.outputs = tf.nn.relu(rup5_2.outputs)
        uconv5_1 = Conv2d(rup5_2, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,name='uconv5_1')
        ruconv5_1 = uconv5_1
        ruconv5_1.outputs = tf.nn.relu(ruconv5_1.outputs)
        uconv5_2 = Conv2d(ruconv5_1, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='uconv5_2')
        print('uconv5_2 : ', uconv5_2.outputs)
        uconv5_3 = uconv5_2
        # uconv5_3 = upsep2_fn(up5_2, uconv5_2, 7)
        print(uconv5_3.outputs)

        ##  up4
        ruconv5_3 = uconv5_3
        ruconv5_3.outputs = tf.nn.relu(ruconv5_3.outputs)
        # up4_0 = UpSampling2dLayer(ruconv5_3, size=(2, 2), is_scale=True, method=1, name='up4_0')
        # up4_1 = Conv2d(up4_0, 128, (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
        #                  name='up4_1')
        up4_1 = DeConv2d(ruconv5_3, 128, (2, 2), strides=(2, 2),
                         padding='SAME', W_init=w_init, b_init=b_init, name='up4_1')
        # print('up4 : ', up4_0.outputs)
        up4_2 = ConcatLayer([up4_1, conv4_3], concat_dim=3, name='up4_2')
        print('concat4 : ', up4_2.outputs)
        rup4_2 = up4_2
        rup4_2.outputs = tf.nn.relu(rup4_2.outputs)
        uconv4_1 = Conv2d(rup4_2, 128, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                          name='uconv4_1')
        ruconv4_1 = uconv4_1
        ruconv4_1.outputs = tf.nn.relu(ruconv4_1.outputs)
        uconv4_2 = Conv2d(ruconv4_1, 128, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                          name='uconv4_2')
        print('uconv4_2 : ', uconv4_2.outputs)
        uconv4_3 = uconv4_2
        # uconv4_3 = upsep2_fn(up4_2, uconv4_2, 8)
        print(uconv4_3.outputs)

        ##  up3
        ruconv4_3 = uconv4_3
        ruconv4_3.outputs = tf.nn.relu(ruconv4_3.outputs)
        # up3_0 = UpSampling2dLayer(ruconv4_3, size=(2, 2), is_scale=True, method=1, name='up3_0')
        # up3_1 = Conv2d(up3_0, 64, (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
        #                  name='up3_1')
        up3_1 = DeConv2d(ruconv4_3, 64, (2, 2),  strides=(2, 2), padding='SAME',
                         W_init=w_init, b_init=b_init, name='up3_1')
        # print('up3 : ', up3_0.outputs)
        up3_2 = ConcatLayer([up3_1, conv3_3], concat_dim=3, name='up3_2')
        print('concat3 : ', up3_2.outputs)
        rup3_2 = up3_2
        rup3_2.outputs = tf.nn.relu(rup3_2.outputs)
        uconv3_1 = Conv2d(rup3_2, 64, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                          name='uconv3_1')
        print('uconv3_1 : ', uconv3_1.outputs)
        ruconv3_1 = uconv3_1
        ruconv3_1.outputs = tf.nn.relu(ruconv3_1.outputs)
        uconv3_2 = Conv2d(ruconv3_1, 64, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                          name='uconv3_2')
        print('uconv3_2 : ', uconv3_2.outputs)
        uconv3_3 = uconv3_2
        # uconv3_3 = upsep2_fn(up3_2, uconv3_2, 9)
        print(uconv3_3.outputs)

        ##  up2
        ruconv3_3 = uconv3_3
        ruconv3_3.outputs = tf.nn.relu(ruconv3_3.outputs)
        # up2_0 = UpSampling2dLayer(ruconv3_3, size=(2, 2), is_scale=True, method=1, name='up2_0')
        # up2_1 = Conv2d(up2_0, 32, (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
        #                  name='up2_1')
        up2_1 = DeConv2d(ruconv3_3, 32, (8, 8), strides=(2, 2),
                         padding='SAME', W_init=w_init, b_init=b_init, name='up2_1')
        # print('up2 : ', up2_0.outputs)
        up2_2 = ConcatLayer([up2_1, conv2_3], concat_dim=3, name='up2_2')
        print('concat2 : ', up2_2.outputs)
        rup2_2 = up2_2
        rup2_2.outputs = tf.nn.relu(rup2_2.outputs)
        uconv2_1 = Conv2d(rup2_2, 32, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                          name='uconv2_1')
        print('uconv2_1 : ', uconv2_1.outputs)
        ruconv2_1 = uconv2_1
        ruconv2_1.outputs = tf.nn.relu(ruconv2_1.outputs)
        uconv2_2 = Conv2d(ruconv2_1, 32, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                          name='uconv2_2')
        print('uconv2_2 : ', uconv2_2.outputs)
        uconv2_3 = uconv2_2
        # uconv2_3 = upsep2_fn(up2_2, uconv2_2, 10)
        print(uconv2_3.outputs)

        ruconv2_3 = uconv2_3
        ruconv2_3.outputs = tf.nn.relu(ruconv2_3.outputs)
        uconv1_1 = Conv2d(ruconv2_3, 3, (3, 3), act=tf.nn.tanh, name='uconv1_1')
        print(" * Output: %s" % uconv1_1.outputs)
        return uconv1_1

def separate_fn(x, y, i):
    #return x
    # x = convolution layer
    # y = pooling layer
    # i = layer index number
    layers = x.outputs.shape[3]
    x_1 = x.outputs[:, :, :, :layers // 2]
    x_2 = x.outputs[:, :, :, layers // 2:layers]
    x1 = InputLayer(x_1, name='input{}_11'.format(i))
    x2 = InputLayer(x_2, name='input{}_12'.format(i))
    #adds = tf.add(x_1, y.outputs, name='adds{}'.format(i))
    adds = ElementwiseLayer([x1, y], tf.add, name='add{}'.format(i))
    #outl = tf.concat([adds, x_2], 3, name='outl{}'.format(i))
    print("shape of adds",adds)
    concat = ConcatLayer([adds, x2], concat_dim=3, name='concat{}'.format(i))
    print("shape of concat", concat)
    #conv = InputLayer(outl, name='conv{}_3'.format(i))
    return concat

def upseparate_fn(x, y, i):
    #return x
    # x = big layer
    # y = small layer
    layers = x.outputs.shape[3]
    #lists = tl.layers.UnStackLayer(x, axis=3, name='lists{}'.format(i))
    #lists1 = tl.layers.StackLayer(lists[:layers // 2], 3, name='lists1_{}'.format(i))
    #adds = tl.layers.ElementwiseLayer(layer=[y, lists1], combine_fn=tf.add, name='adds{}'.format(i))
    x_1 = x.outputs[:, :, :, :layers // 2]
    x1 = InputLayer(x_1, name='input{}_2'.format(i))
    #add = tf.add(x_1, y.outputs, name='upadds{}'.format(i))
    adds = ElementwiseLayer([x1, y], tf.add, name='add{}_2'.format(i))
    #adds = InputLayer(add, 'up{}_3'.format(i))
    return adds

def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        #conv = network
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        #conv = network
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
        conv = network
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv

def deconv_layer(input_layer, sz, str, alpha, is_training=False):
    scale = 2

    filter_size = (2 * scale - scale % 2)
    num_in_channels = int(sz[3])
    num_out_channels = int(sz[4])

    # create bilinear weights in numpy array
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                   (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_out_channels, num_in_channels))
    for i in range(num_out_channels):
        weights[:, :, i, i] = bilinear_kernel

    init_matrix = tf.constant_initializer(value=weights, dtype=tf.float32)

    network = tl.layers.DeConv2dLayer(input_layer,
                                shape = [filter_size, filter_size, num_out_channels, num_in_channels],
                                output_shape = [sz[0], sz[1]*scale, sz[2]*scale, num_out_channels],
                                strides=[1, scale, scale, 1],
                                W_init=init_matrix,
                                padding='SAME',
                                act=tf.identity,
                                name=str)

    network = tl.layers.BatchNormLayer(network, is_train=is_training, name='%s/batch_norm_dc'%str)
    network.outputs = tf.maximum(alpha*network.outputs, network.outputs, name='%s/leaky_relu_dc'%str)

    return network

