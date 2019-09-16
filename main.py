#! /usr/bin/python
# -*- coding: utf8 -*-
import math
import pickle, random
import time
import imageio as ii
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_d, Vgg19_simple_api, SRGAN_g3, HDR_C_GAN_d, unet, Temp_d
from utils import *
from config import config, log_config
from psnr import *

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
n_epoch_init = config.TRAIN.n_epoch_init
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size*config.Size.num_sub_imgs))
sub_img_size = config.Size.sub_img_size
with_vgg = config.TRAIN.with_vgg
with_mse = config.TRAIN.with_mse

def train():
    import os, time
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    num_sub_imgs = config.Size.num_sub_imgs

    input_img_path = config.TRAIN.input_img_path
    if config.TRAIN.input_type == "quan":
        input_img_path = config.TRAIN.input_img_path + "_quan"
    elif config.TRAIN.input_type == "clip":
        input_img_path = config.TRAIN.input_img_path + "_clip"
    else:
        print("input_type error")
        return

    label_img_path = config.TRAIN.label_img_path

    ###====================== PRE-LOAD DATA ===========================###
    train_input_img_list = sorted(
        tl.files.load_file_list(path=input_img_path, regx='.*.png', printable=False))
    train_label_img_list = sorted(
        tl.files.load_file_list(path=label_img_path, regx='.*.png', printable=False))
    print('train_input_img_list : ', train_input_img_list)
    print('train_label_img_list : ', train_label_img_list)

    ## If your machine have enough memory, please pre-load the whole train set.
    train_input_imgs = tl.vis.read_images(train_input_img_list, path=input_img_path, n_threads=32)
    train_label_imgs = tl.vis.read_images(train_label_img_list, path=label_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32',
                             [batch_size * num_sub_imgs, sub_img_size, sub_img_size, 3],
                             name='t_image_input_to_generator')
    t_target_image = tf.placeholder('float32',
                                    [batch_size * num_sub_imgs, sub_img_size, sub_img_size, 3],
                                    name='t_target_image')

    net_g = unet(t_image, reuse=False)

    net_g.print_params(False)
    net_g.print_layers()

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    if with_vgg:
        t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0,align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
        t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0,align_corners=False)  # resize_generate_image_for_vgg

        net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
        _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

    ## test inference
    net_g_test = unet(t_image, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================##
    if with_mse:
        mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
        if with_vgg:
            vgg_loss = 5e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
            mse_loss = mse_loss + vgg_loss
    else:
        if with_vgg:
            mse_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)


    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)


    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/checkpoint.npz',network=net_g)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"

    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()

    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    if with_vgg:
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        tl.files.assign_params(sess, params, net_vgg)
        net_vgg.print_params(False)
        net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    _sample_imgs_input = train_input_imgs[0:batch_size]
    _sample_imgs_label = train_label_imgs[0:batch_size]
    sample_imgs_input, sample_imgs_label = crop_sub_imgs(_sample_imgs_input, _sample_imgs_label)

    tl.vis.save_images(sample_imgs_input, [ni, ni], save_dir_ginit + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_label, [ni, ni], save_dir_ginit + '/_train_sample_384.png')

    ###========================= initialize G ====================###
    ## fixed learning rate
    lr_g = config.TRAIN.lr_g

    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)

    with open("loss.txt", 'w') as f:
        f.write("")
    decay_g = config.TRAIN.decay_g
    init_time = time.time()
    for epoch in range(0, n_epoch_init + 1):
        if epoch != 0 and (epoch % decay_g == 0):
            new_lr_decay = lr_decay ** (epoch // decay_g)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        for idx in range(0, len(train_input_imgs), batch_size):
            step_time = time.time()
            b_imgs_input, b_imgs_label = list_sub_imgs(train_input_imgs[idx:idx + batch_size],
                                                       train_label_imgs[idx:idx + batch_size])
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_input, t_target_image: b_imgs_label})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
                epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
            epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)
        with open("loss.txt", 'a') as f:
            f.write(str(total_mse_loss / n_iter) + "\n")

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs,
                           {t_image: sample_imgs_input})  # ; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']),
                              sess=sess)

    print("init complete %dsec" % (init_time - time.time()))


def test(i=0):
    ## create folders to save result images
    # 0~7
    # i = 0
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"
    input_img_path = config.Test.input_img_path
    if config.TRAIN.input_type == "quan":
        input_img_path = config.Test.input_img_path + "_quan"
    elif config.TRAIN.input_type == "clip":
        input_img_path = config.Test.input_img_path + "_clip"
    else:
        print("input_type error")
        return

    label_img_path = config.Test.label_img_path
    ###====================== PRE-LOAD DATA ===========================###
    img = np.asarray(ii.imread(input_img_path + os.sep + "input_{:04}_clip.png".format(i), 'PNG-FI'))
    target = np.asarray(ii.imread(label_img_path + os.sep + "label_{:04}.png".format(i), 'PNG-FI'))
    img = (img / (255./2.) ) -1
    target = (target / (255. / 2.)) - 1

    mul = 16
    size = img.shape
    h, w, d = img.shape
    print(h, w)
    h = (h // mul + 1) * mul
    w = (w // mul + 1) * mul

    img_process = np.zeros(shape=(h, w, 3))
    img_process[:size[0], :size[1], :] = img
    print(img_process.shape)

    print(img.min(), img.max())

    tl.vis.save_image(img, save_dir + '/test_input_{0:03d}.png'.format(i))
    tl.vis.save_image(target, save_dir + '/test_label_{0:03d}.png'.format(i))

    ###========================== DEFINE MODEL ============================###

    t_image = tf.placeholder('float32', [1, h, w, d], name='input_image')
    net_g = unet(t_image, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/checkpoint.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [img_process]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("input size: %s /  generated label size: %s" % (size, out.shape))
    print("[*] {0:03d}th save image".format(i))

    output = out[0, :size[0], :size[1], :]
    print(output.min(), output.max())
    print(output.shape)
    tl.vis.save_image(output, save_dir + '/test_gen_{0:03d}.png'.format(i))

    print("Test complete")

    ###evaluate psnr
    img_array = np.asarray(target, dtype=np.float32)
    img_array /= np.max(img_array) * 1.0
    _output = output / output.max() * 1.0
    psnr = psnr_func(_output, img_array)
    print("%03dth img psnr :" % i, psnr)
    with open("psnr.txt", "a") as f:
        f.write("%03dth img psnr :" % i + str(psnr) + "\n")

if __name__ == '__main__':

    # with tf.device("/gpu:0"):
    import argparse

    test_idx = 9

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate, test')
    #parser.add_argument('--mode', type=str, default='test', help='srgan, evaluate,test')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'train':
        train()
    elif tl.global_flag['mode'] == 'test':
        test(test_idx)
    else:
        raise Exception("Unknow --mode")

