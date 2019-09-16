import tensorlayer as tl
import os
import imageio as ii
import numpy as np
import math
from utils import unet_fn

def psnr():
    import numpy as np
    import math
    train_img_list = sorted(
        tl.files.load_file_list(path="samples", regx='.*.png', printable=False))
    print("train_img_list : ",train_img_list)
    for idx, img_name in enumerate(train_img_list[:len(train_img_list)//2]):
        out_image = ii.imread("samples" + os.sep + train_img_list[idx], "PNG-FI")
        label_image =ii.imread("samples" + os.sep + train_img_list[idx+(len(train_img_list)//2)], "PNG-FI")
        im1 = unet_fn(out_image) #/ 255.0
        im2 = unet_fn(label_image) #/ 255.0
        # im1 = tf.image.convert_image_dtype(in_img1, tf.float32)
        # im2 = tf.image.convert_image_dtype(in_img2, tf.float32)
        mse = np.mean((im1-im2)**2)
        if mse == 0:
            return 100
        PIXEL_MAX = 2.0

        psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        print(img_name + "의 psnr = " + str(psnr))

def psnr_func(input,label):
    input=input/np.max(input)
    label=label/np.max(label)
    mse = np.mean((input - label) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 2.0 #1.0

    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    return psnr

if __name__ == "__main__":
    psnr()