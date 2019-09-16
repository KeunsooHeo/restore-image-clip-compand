from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.lr_g = 1e-4 #0.0002
config.TRAIN.beta1 = 0.9 #0.5

## initialize G
config.TRAIN.n_epoch_init = 0
config.TRAIN.lr_decay = 0.8
config.TRAIN.decay_g = int(config.TRAIN.n_epoch_init / 2)
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 10)
config.TRAIN.with_mse = True
config.TRAIN.with_vgg = True
config.TRAIN.input_type = "clip"


## train set location
#config.TRAIN.hr_img_path = 'data2017/DIV2K_train_HR/'
#config.TRAIN.lr_img_path = 'data2017/DIV2K_train_LR_bicubic/X4/'
config.TRAIN.hdr_img_path = "MyHDRdatabase/Train_HDRimages"
config.TRAIN.ldr_img_path = "LDRDB/Train_MSCoCo2000"
config.TRAIN.input_img_path = "Train/Input"
config.TRAIN.label_img_path = "Train/Label"
config.TRAIN.n_fixed_patch = 0

config.VALID = edict()
## test set location
config.Test = edict()
config.Test.hdr_img_path = "MyHDRdatabase/Test_HDRimages"
config.Test.ldr_img_path = "LDRDB/TestImage"
config.Test.input_img_path = "Test/Input"
config.Test.label_img_path = "Test/Label"

config.Size = edict()
config.Size.sub_img_size = 128 #minimum 64
config.Size.num_sub_imgs = 1

config.Model = edict()
config.Model.model = "unet"
#config.Model.model = "none" #  #

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")

