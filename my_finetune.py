#%%
from __future__ import print_function, unicode_literals
#%%
import sys
import json

sys.path.append(".")
import argparse
from tqdm import tqdm
import torch
import subprocess
import numpy as np
import os

from src.models.my_wrap import Wrap_Resnet
from src.models.unsupervised.hybrid2_model import Hybrid2Model
from testing.fh_utils import (
    json_load,
    db_size,
    get_bbox_from_pose,
    read_img,
    convert_order,
    move_palm_to_wrist,
    modify_bbox,
    preprocess,
    create_affine_transform_from_bbox,
)

BBOX_SCALE = 0.33
CROP_SIZE = 224
# DS_PATH = "/hdd/Datasets/freihand_dataset/"
DS_PATH = "/home/dlc/cll/peclr/data/raw/freihand_dataset"
#%%
from pprint import pformat
from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger
from src.constants import (
    COMET_KWARGS,
    HYBRID2_CONFIG,
    SIMCLR_CONFIG ,
    BASE_DIR,
    TRAINING_CONFIG_PATH,
)
from src.data_loader.data_set_cbg import Data_Set
from src.data_loader.utils import get_data_cbg, get_train_val_split

from src.experiments.utils import (
    get_callbacks,
    get_general_args,
    get_model,
    prepare_name,
    save_experiment_key,
    update_model_params,
    update_train_params,
)
from src.utils import get_console_logger, read_json
experiment_type = "hybrid2"
# experiment_type = "simclr"
console_logger = get_console_logger(__name__)
args=[ 
                "--color_jitter",
                "--random_crop",
                "--rotate", 
                "--crop",
                "-resnet_size", "50",  
                "-sources", "freihand", 
                "--resize",   
                "-epochs", "500", 
                "-batch_size","128",  
                "-accumulate_grad_batches", "16", 
                "-save_top_k", "1",  
                "-save_period", "1",   
                "-num_workers", "8"
                # // "-d", "./kodim",
                # // "--epochs","300", 
                # // "-lr", "1e-4", 
                # // "--batch-size", "1",
                # // "--cuda", "--save",
                # // "--test-batch-size", "1"
            ]
# get_general_args("Hybrid model 2 training script.")
with open('args128.json','r+') as file:
    content=file.read()
    
content=json.loads(content)
args=argparse.Namespace(**content)
# args = get_general_args("Simclr model training script.")
train_param = edict(read_json(TRAINING_CONFIG_PATH))
train_param = update_train_params(args, train_param)
# model_param_path = SIMCLR_CONFIG
model_param_path = HYBRID2_CONFIG  # SIMCLR_CONFIG
model_param = edict(read_json(model_param_path))
console_logger.info(f"Train parameters {pformat(train_param)}")
seed_everything(train_param.seed)
data = get_data_cbg(
        Data_Set, train_param, sources=args.sources, experiment_type=experiment_type
    )
model_param = update_model_params(model_param, args, len(data), train_param)
model_param.augmentation = [
key for key, value in train_param.augmentation_flags.items() if value
]
console_logger.info(f"Model parameters {pformat(model_param)}")
model = get_model(
experiment_type="hybrid2",#"simclr"
heatmap_flag=args.heatmap,
denoiser_flag=args.denoiser,
)(config=model_param)
model=Hybrid2Model(config=model_param)

model_path = '/home/zlc/cll/code/peclr_cbg/data/models/hybrid2-frei-bs256/epoch=286.ckpt'
dev = torch.device("cuda")
checkpoint=torch.load(model_path)
model=Hybrid2Model(config=model_param)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
#%%
# import cv2
# img_name = "/home/zlc/cll/code/peclr/data/raw/freihand_dataset/training/rgb/00000000.jpg"
# img = cv2.cvtColor(cv2.imread(img_name),cv2.COLOR_BGR2RGB)

# import matplotlib.pyplot as plt
# plt.imshow(img)
# trafo = lambda x: np.transpose(x[:, :, ::-1], [2, 0, 1]).astype(np.float32) / 255.0 - 0.5
# img_t = trafo(img)
# batch = torch.Tensor(np.stack([img_t], 0)).cuda()

#%%
from src.data_loader.data_set_cbg import Data_Set
from torchvision import transforms
from torch.utils.data import DataLoader
train_param = edict(
    read_json("./src/experiments/config/training_config.json")
)
train_param.augmentation_flags.resize = True
train_param.augmentation_flags.random_crop = True
train_data = Data_Set(
    config=train_param,
    transform=transforms.ToTensor(),
    split="train",
    experiment_type="supervised",
    source="freihand",
)
train_data.is_training=False
# print(da)
data_loader = DataLoader(train_data, batch_size=128, num_workers=4)
# data = get_data_cbg(
#         Data_Set, train_param, sources=args.sources, experiment_type="supervised"
#     )

#%%
emb=[]
print(len(data_loader))
for i in enumerate(data_loader):
    # print(len(i))
    # print(i[1].keys())
    # break
    print(len(i[1]['image']))
    tmp=model.get_encodings(i[1]['image'])
    emb+=tmp

