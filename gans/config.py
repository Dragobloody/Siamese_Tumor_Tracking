from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 32 # [16] use 8 if your GPU memory is small, and use [2, 4] in tl.vis.save_images / use 16 for faster training
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.5

## initialize G
config.TRAIN.n_epoch_init = 51
    
## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 200
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 10)




def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
