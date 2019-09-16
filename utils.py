import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
from config import config, log_config
import imageio as ii
import time
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np
import random as r
import os

thrd_persentage = 96.0


def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')


def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=config.Size.hr_img_size, hrg=config.Size.hr_img_size, is_random=is_random)
    x = x / 255
    return x


def crop_sub_imgs(input, label):
    sub_size = config.Size.sub_img_size
    batch_size = config.TRAIN.batch_size
    r_input_list = []
    r_label_list = []
    for i in range(batch_size):
        w, h, _ = input[i].shape
        input[i] = (input[i] / (255. / 2.)) - 1
        label[i] = (label[i] / (255. / 2.)) - 1
        rw, rh = r.randrange(w - sub_size), r.randrange(h - sub_size)
        r_input_list.append(input[i][rw:rw + sub_size, rh:rh + sub_size])
        r_label_list.append(label[i][rw:rw + sub_size, rh:rh + sub_size])

    return np.asarray(r_input_list), np.asarray(r_label_list)


def list_sub_imgs(input, label):
    import random as r
    sub_size = config.Size.sub_img_size
    # input = input / 255. #normalize
    num_sub_imgs = config.Size.num_sub_imgs

    num_fixed_patch = config.TRAIN.n_fixed_patch
    if len(input) != len(label):
        print("invaild len")
        return

    r_input_list = []
    r_num_list = []
    for img in input:
        img = img / 255
        w, h, _ = img.shape
        nx = w // sub_size + 1
        ny = h // sub_size + 1

        zeros = np.zeros(shape=(sub_size * nx, sub_size * ny, 3))
        zeros[:w, :h, :] = img
        list_imgs = []
        for i in range(nx):
            for j in range(ny):
                list_imgs.append(zeros[sub_size * i:sub_size * (i + 1), sub_size * j:sub_size * (j + 1):, :])

        r_num = r.randrange(nx * ny)
        temp_r_num_list = []
        temp_list_imgs = list_imgs.copy()

        # 고정추출
        for x in range(num_fixed_patch):
            maxidx = 0
            max_value = 0.
            for i in range(len(temp_list_imgs)):
                array = temp_list_imgs[i]
                aver = np.average(np.asarray(array))
                if max_value <= aver:
                    max_value = aver
                    maxidx = i
            temp_r_num_list.append(maxidx)
            temp_list_imgs[maxidx] = temp_list_imgs[maxidx] * 0

        for i in range(num_sub_imgs - num_fixed_patch):
            while r_num in temp_r_num_list:
                r_num = r.randrange(nx * ny)
            temp_r_num_list.append(r_num)
        r_num_list.append(temp_r_num_list)

        for i in temp_r_num_list:
            r_input_list.append(list_imgs[i])

    r_label_list = []
    for idx, img in enumerate(label):
        img = img / 255
        w, h, _ = img.shape
        nx = w // sub_size + 1
        ny = h // sub_size + 1

        zeros = np.zeros(shape=(sub_size * nx, sub_size * ny, 3))
        zeros[:w, :h, :] = img
        list_imgs = []
        for i in range(nx):
            for j in range(ny):
                list_imgs.append(zeros[sub_size * i:sub_size * (i + 1), sub_size * j:sub_size * (j + 1):, :])

        for i in r_num_list[idx]:
            r_label_list.append(list_imgs[i])

    return np.asarray(r_input_list), np.asarray(r_label_list)


def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[config.Size.sub_img_size, config.Size.sub_img_size], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def unet_fn(x):
    # x = x / 255.
    x = x / (255. / 2.)
    x = x - 1.
    return x


def simulate_hdr():
    ##### set up parameter
    rbit_hsat = 2
    scale_hsat = int(pow(2, rbit_hsat))
    hfact = 5.0  # 9.0 #10.0
    clip_thres = 170  # 200 250
    '''
    ##### image set up
    train_label_img_list = sorted(
        tl.files.load_file_list(path=config.TRAIN.ldr_img_path, regx='.*.png', printable=False))
    print('train_label_img_list : ', train_label_img_list)

    for idx, img_name in enumerate(train_label_img_list):
        start_time = time.time()
        inimage = ii.imread(config.TRAIN.ldr_img_path+os.sep+img_name, 'PNG-FI')
        h,w,z = inimage.shape
        if(z != 3):
            print("%d img has %d demension"%(idx,z))
            continue

        in_img1 = pow(inimage/255.0, 2.2) # linearization

        # quantization
        in_img2 = pow(in_img1, 1.0 / hfact)
        in_img2 = np.uint8(in_img2 * 255)
        out_hsat = in_img2 // scale_hsat # quantization for high value
        out_hsat = out_hsat * scale_hsat  # quantization for high value
        out_hsat = out_hsat / 255.0
        out_hsat = pow(out_hsat, hfact)  # restoration foe compressed high valuer
        out_hsat = pow(out_hsat, 1.0 / 2.2)  # gamma correction

        print("%04d/%04d quantization complete" % (idx+1,len(train_label_img_list)))
        # clipping high value
        out_clip = np.zeros(inimage.shape)
        in_img2 = np.uint8(in_img1 * 255)
        for i in range(h):
            for j in range(w):
                for k in range(z):
                    if (in_img2[i,j,k] > clip_thres):
                        out_clip[i,j, k] = clip_thres
                    else:
                        out_clip[i, j, k] = in_img2[i,j,k]
        out_clip = out_clip / clip_thres
        out_clip = pow(out_clip, 1.0/2.2)  # gamma correction
        print("%04d/%04d clipping complete" % (idx + 1, len(train_label_img_list)))

        tl.files.exists_or_mkdir(config.TRAIN.label_img_path)
        tl.files.exists_or_mkdir(config.TRAIN.input_img_path+"_quan")
        tl.files.exists_or_mkdir(config.TRAIN.input_img_path+"_clip")

        ii.imwrite(config.TRAIN.label_img_path+os.sep+"label_%04d.png"%idx, inimage, 'PNG-FI')
        ii.imwrite(config.TRAIN.input_img_path+"_quan"+os.sep+"input_%04d_quan.png"%idx, out_hsat, 'PNG-FI')
        ii.imwrite(config.TRAIN.input_img_path+"_clip"+os.sep+"input_%04d_clip.png"%idx, out_clip, 'PNG-FI')
        print("%04d/%04d saved " % (idx + 1, len(train_label_img_list)))
        print("time %04.1fsec"%(time.time()-start_time))
    '''
    #############################
    #####Test SET################
    #############################

    ##### image set up
    train_label_img_list = sorted(
        tl.files.load_file_list(path=config.Test.ldr_img_path, regx='.*.png', printable=False))
    print('train_label_img_list : ', train_label_img_list)

    for idx, img_name in enumerate(train_label_img_list):
        start_time = time.time()
        inimage = ii.imread(config.Test.ldr_img_path + os.sep + img_name, 'PNG-FI')
        h, w, z = inimage.shape
        in_img1 = pow(inimage / 255.0, 2.2)  # linearization

        # quantization
        in_img2 = pow(in_img1, 1.0 / hfact)
        in_img2 = np.uint8(in_img2 * 255)
        out_hsat = in_img2 // scale_hsat  # quantization for high value
        out_hsat = out_hsat * scale_hsat  # quantization for high value
        out_hsat = out_hsat / 255.0
        out_hsat = pow(out_hsat, hfact)  # restoration foe compressed high valuer
        out_hsat = pow(out_hsat, 1.0 / 2.2)  # gamma correction

        print("%04d/%04d quantization complete" % (idx + 1, len(train_label_img_list)))
        # clipping high value
        out_clip = np.zeros(inimage.shape)
        in_img2 = np.uint8(in_img1 * 255)
        for i in range(h):
            for j in range(w):
                for k in range(z):
                    if (in_img2[i, j, k] > clip_thres):
                        out_clip[i, j, k] = clip_thres
                    else:
                        out_clip[i, j, k] = in_img2[i, j, k]
        out_clip = out_clip / clip_thres
        out_clip = pow(out_clip, 1.0 / 2.2)  # gamma correction
        print("%04d/%04d clipping complete" % (idx + 1, len(train_label_img_list)))

        tl.files.exists_or_mkdir(config.Test.label_img_path)
        tl.files.exists_or_mkdir(config.Test.input_img_path + "_quan")
        tl.files.exists_or_mkdir(config.Test.input_img_path + "_clip")

        ii.imwrite(config.Test.label_img_path + os.sep + "label_%04d.png" % idx, inimage, 'PNG-FI')
        ii.imwrite(config.Test.input_img_path + "_quan" + os.sep + "input_%04d_quan.png" % idx, out_hsat, 'PNG-FI')
        ii.imwrite(config.Test.input_img_path + "_clip" + os.sep + "input_%04d_clip.png" % idx, out_clip, 'PNG-FI')
        print("%04d/%04d saved " % (idx + 1, len(train_label_img_list)))
        print("time %4.1fsec" % (time.time() - start_time))


def preprocess_trainset():
    import os, math, time

    train_hdr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hdr_img_path, regx='.*.hdr', printable=False))
    train_exr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hdr_img_path, regx='.*.exr', printable=False))
    print("train_hdr_img_list", train_hdr_img_list)
    print("train_exr_img_list", train_exr_img_list)
    # imghdr = tl.vis.read_images(train_hdr_img_list, path=config.TRAIN.hdr_img_path, n_threads=32)
    thrd_persentage = 96.0
    n = 0
    for filename in train_hdr_img_list:
        imghdr = ii.imread(config.TRAIN.hdr_img_path + os.sep + filename, 'HDR-FI')

        step_time = time.time()
        # print("imghdr lens : ", len(imghdr))
        # print(train_hdr_img_list)

        label_ = np.zeros(imghdr.shape)
        input_ = np.zeros(imghdr.shape)
        h, w, z = imghdr.shape

        a = np.sort(imghdr, axis=None)
        thres = a[int((3 * h * w - 1) / (100.0 / thrd_persentage))]  # 95% of max
        h, w, z = imghdr.shape

        m = a.max()
        if (m < thres * 3.):
            print("one of images was excluded")
            continue

        for i in range(h):
            for j in range(w):
                for k in range(z):
                    if (imghdr[i, j, k] <= thres):
                        input_[i, j, k] = imghdr[i, j, k]
                    else:
                        input_[i, j, k] = thres
        input_ = input_ / np.max(input_)
        input_ = np.power(input_, 1.0 / 2.0)
        print(n + 1, "input complete")

        thres = thres / 4
        for i in range(h):
            for j in range(w):
                for k in range(z):
                    if (imghdr[i, j, k] <= thres * 1.0):
                        label_[i, j, k] = 0.5 * math.sqrt(imghdr[i, j, k] / thres)
                    else:
                        if (imghdr[i, j, k] <= thres * 12.0):
                            label_[i, j, k] = 0.17883277 * math.log(imghdr[i, j, k] / thres - 0.28466892) + 0.55991073
                        else:
                            label_[i, j, k] = 0.17883277 * math.log(12.0 - 0.28466892) + 0.55991073
        print(n + 1, "label complete")
        tl.files.exists_or_mkdir(config.TRAIN.input_img_path)
        tl.files.exists_or_mkdir(config.TRAIN.label_img_path)
        ii.imwrite(config.TRAIN.input_img_path + os.sep + "input_{0:03d}.png".format(n + 1), input_, format='PNG-PIL')
        ii.imwrite(config.TRAIN.label_img_path + os.sep + "label_{0:03d}.png".format(n + 1), label_, format='PNG-PIL')
        print(n + 1, "saving...")

        print("HDR : %d/%d process complete " % (n + 1, len(train_hdr_img_list)), "time :", time.time() - step_time)
        n = n + 1

    n0 = n
    for filename in train_exr_img_list:

        imghdr = ii.imread(config.TRAIN.hdr_img_path + os.sep + filename, 'EXR-FI')

        step_time = time.time()
        # print("imghdr lens : ", len(imghdr))
        # print(train_hdr_img_list)

        label_ = np.zeros(imghdr.shape)
        input_ = np.zeros(imghdr.shape)
        h, w, z = imghdr.shape

        a = np.sort(imghdr, axis=None)
        thres = a[int((3 * h * w - 1) / (100.0 / thrd_persentage))]  # 95% of max
        h, w, z = imghdr.shape

        m = a.max()
        if (m < thres * 3.):
            print("one of images was excluded")
            continue

        for i in range(h):
            for j in range(w):
                for k in range(z):
                    if (imghdr[i, j, k] <= thres):
                        input_[i, j, k] = imghdr[i, j, k]
                    else:
                        input_[i, j, k] = thres
        input_ = input_ / np.max(input_)
        input_ = np.power(input_, 1.0 / 2.0)
        print(n + 1 + n0, "input complete")

        thres = thres / 4
        for i in range(h):
            for j in range(w):
                for k in range(z):
                    if (imghdr[i, j, k] <= thres * 1.0):
                        label_[i, j, k] = 0.5 * math.sqrt(imghdr[i, j, k] / thres)
                    else:
                        if (imghdr[i, j, k] <= thres * 12.0):
                            label_[i, j, k] = 0.17883277 * math.log(imghdr[i, j, k] / thres - 0.28466892) + 0.55991073
                        else:
                            label_[i, j, k] = 0.17883277 * math.log(12.0 - 0.28466892) + 0.55991073
        print(n + 1, "label complete")
        tl.files.exists_or_mkdir(config.TRAIN.input_img_path)
        tl.files.exists_or_mkdir(config.TRAIN.label_img_path)
        ii.imwrite(config.TRAIN.input_img_path + os.sep + "input_{0:03d}.png".format(n + 1), input_, format='PNG-PIL')
        ii.imwrite(config.TRAIN.label_img_path + os.sep + "label_{0:03d}.png".format(n + 1), label_, format='PNG-PIL')
        print(n + 1, "saving...")

        print("EXR : %d/%d process complete " % (n + 1, len(train_exr_img_list) + n0), "time :",
              time.time() - step_time)
        n = n + 1

    print("processing complete")


def preprocess_testset():
    import os, math, time
    test_hdr_img_list = sorted(tl.files.load_file_list(path=config.Test.hdr_img_path, regx='.*.hdr', printable=False))
    print("test_hdr_img_list", test_hdr_img_list)
    # imghdr = tl.vis.read_images(test_hdr_img_list, path=config.Test.hdr_img_path, n_threads=32)
    thrd_persentage = 96.0
    for n, filename in enumerate(test_hdr_img_list):
        imghdr = ii.imread(config.Test.hdr_img_path + os.sep + filename, 'HDR-FI')
        # print("imghdr lens : ",len(imghdr))
        # print(test_hdr_img_list)
        step_time = time.time()

        label_ = np.zeros(imghdr.shape)
        input_ = np.zeros(imghdr.shape)
        h, w, z = imghdr.shape

        a = np.sort(imghdr, axis=None)
        thres = a[int((3 * h * w - 1) / (100.0 / thrd_persentage))]  # 95% of max
        h, w, z = imghdr.shape

        for i in range(h):
            for j in range(w):
                for k in range(z):
                    if (imghdr[i, j, k] <= thres):
                        input_[i, j, k] = imghdr[i, j, k]
                    else:
                        input_[i, j, k] = thres
        input_ = input_ / np.max(input_)
        input_ = np.power(input_, 1.0 / 2.0)
        print(n + 1, "imput complete")

        thres = thres / 4
        for i in range(h):
            for j in range(w):
                for k in range(z):
                    if (imghdr[i, j, k] <= thres * 1.0):
                        label_[i, j, k] = 0.5 * math.sqrt(imghdr[i, j, k] / thres)
                    else:
                        if (imghdr[i, j, k] <= thres * 12.0):
                            label_[i, j, k] = 0.17883277 * math.log(imghdr[i, j, k] / thres - 0.28466892) + 0.55991073
                        else:
                            label_[i, j, k] = 0.17883277 * math.log(12.0 - 0.28466892) + 0.55991073
        print(n + 1, "label complete")
        tl.files.exists_or_mkdir(config.Test.input_img_path)
        tl.files.exists_or_mkdir(config.Test.label_img_path)
        ii.imwrite(config.Test.input_img_path + os.sep + "input_{0:03d}.png".format(n), input_, format='PNG-PIL')
        ii.imwrite(config.Test.label_img_path + os.sep + "label_{0:03d}.png".format(n), label_, format='PNG-PIL')
        print(n + 1, "saving...")
        print("%d/%d process complete " % (n + 1, len(test_hdr_img_list)), "time :", time.time() - step_time)

    print("processing complete")


if __name__ == '__main__':
    with tf.device("/gpu:0"):
        # preprocess_trainset()
        # preprocess_testset()
        simulate_hdr()