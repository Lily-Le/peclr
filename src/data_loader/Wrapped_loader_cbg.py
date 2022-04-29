from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
import os

from scipy.ndimage import gaussian_filter
from torch.tensor import Tensor
from src.data_loader.utils import convert_2_5D_to_3D
from typing import List, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.data_loader.joints import Joints
from src.utils import read_json,json_load
from torch.utils.data import Dataset
import random
from scipy.ndimage.morphology import binary_erosion
import matplotlib.pyplot as plt

BOUND_BOX_SCALE = 0.33
# BG_PIC_PATH = '/home/d3-ai/cll/HanCo/bg_new'
# BG_IND_PATH ='/home/d3-ai/cll/contra-hand/bg_inds.json'
BG_PIC_PATH ='/home/zlc/cll/data/bg_new'
BG_IND_PATH ='/home/zlc/cll/data/bg_inds.json'
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y
def mix(fg_img, mask_fg, bg_img, do_smoothing, do_erosion):
    """ Mix fg and bg image. Keep the fg where mask_fg is True. """
    assert bg_img.shape == fg_img.shape
    fg_img = fg_img.copy()
    mask_fg = mask_fg.copy()
    bg_img = bg_img.copy()

    if len(mask_fg.shape) == 2:
        mask_fg = np.expand_dims(mask_fg, -1)

    if do_erosion:
        mask_fg = binary_erosion(mask_fg, structure=np.ones((5, 5, 1)) )

    mask_fg = mask_fg.astype(np.float32)

    if do_smoothing:
        mask_fg = gaussian_filter(mask_fg, sigma=0.5)

    merged = (mask_fg * fg_img + (1.0 - mask_fg) * bg_img).astype(np.uint8)
    return merged


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)