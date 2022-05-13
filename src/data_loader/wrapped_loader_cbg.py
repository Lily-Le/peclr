from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
import os

from scipy.ndimage import gaussian_filter
from torch.tensor import Tensor
from src.data_loader.utils import convert_2_5D_to_3D
from typing import List, Tuple
from src.data_loader.data_set_cbg import Data_Set_cbg

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
from torchvision import transforms
from src.constants import BG_PIC_PATH,BG_IND_PATH
BG_INDS = json_load(BG_IND_PATH)
BOUND_BOX_SCALE = 0.33
# BG_PIC_PATH = '/home/d3-ai/cll/HanCo/bg_new'
# BG_IND_PATH ='/home/d3-ai/cll/contra-hand/bg_inds.json'
from src.data_loader.data_set_cbg import Data_Set_cbg, F_DB_cbg

transform=transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                                ),
                            ]
                        ),
# def change_bg(
#     self, image: np.array, mask: np.array,joints: JOINTS_25D
# ) -> Tuple[np.array, JOINTS_25D, tuple]:
    
#     rid = random.randint(0, 521)
#     # bg_image_new_path = os.path.join(base_path, 'background_subtraction/background_examples/bg_new/%05d.jpg' % rid)
#     bg_image_new_path = os.path.join(BG_PIC_PATH, BG_INDS[rid])
#     bg_img_new = cv2.cvtColor(cv2.imread(bg_image_new_path),cv2.COLOR_BGR2RGB)

#     # bg_img_new=bg_img_new.copy()
#     h, w = mask.shape[:2]

#     bg_img_new=cv2.resize(bg_img_new, (w,h))#,interpolation=cv2.INTER_CUBIC)
#     # bg_img_new= np.asarray(bg_img_new.resize(fg_img.shape,refcheck=False))
#     # bg_img_new = tmp__
    

#     merged = mix(image, mask, bg_img_new, do_smoothing=True, do_erosion=True)
    

#     return merged, joints

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y
# def change_bg(
#     self, image: np.array, mask: np.array,joints: JOINTS_25D
# ) -> Tuple[np.array, JOINTS_25D, tuple]:
    
#     rid = random.randint(0, 521)
#     # bg_image_new_path = os.path.join(base_path, 'background_subtraction/background_examples/bg_new/%05d.jpg' % rid)
#     bg_image_new_path = os.path.join(BG_PIC_PATH, BG_INDS[rid])
#     bg_img_new = cv2.cvtColor(cv2.imread(bg_image_new_path),cv2.COLOR_BGR2RGB)

#     # bg_img_new=bg_img_new.copy()
#     h, w = mask.shape[:2]

#     bg_img_new=cv2.resize(bg_img_new, (w,h))#,interpolation=cv2.INTER_CUBIC)
#     # bg_img_new= np.asarray(bg_img_new.resize(fg_img.shape,refcheck=False))
#     # bg_img_new = tmp__
    

#     merged = mix(image, mask, bg_img_new, do_smoothing=True, do_erosion=True)
    

#     return merged, joints







'''
 return {
            **{"transformed_image1": img1, "transformed_image2": img2},
            **{"mask1": mask1,"mask2":mask2},
            **{f"{k}_1": v for k, v in param1.items() if v is not None},
            **{f"{k}_2": v for k, v in param2.items() if v is not None},
        }
'''

class WrappedDataLoader:

    def __init__(self, dl):
        self.dl = dl
        # self.func = func
        self.transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                        ),
                    ]
                )
    def __len__(self):
        return len(self.dl)
    
    def mix(self,fg_img, mask_fg, bg_img, do_smoothing, do_erosion):
        """ Mix fg and bg image. Keep the fg where mask_fg is True. """
        assert bg_img.shape == fg_img.shape
        mask_fg = mask_fg/255.
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
   
    def change_batch_background(self,sample,bg_img_new1,bg_img_new2):

        img1=np.array(sample["transformed_image1"])
        img2=np.array(sample["transformed_image2"])

        mask1=np.array(sample['mask1'])
        mask2=np.array(sample['mask2'])
        bg_img_new1=bg_img_new1.copy()
        bg_img_new2=bg_img_new2.copy()
        h, w = mask1[0].shape[:2]

        bg_img_new1=cv2.resize(bg_img_new1, (w,h))#,interpolation=cv2.INTER_CUBIC)
        bg_img_new2=cv2.resize(bg_img_new2, (w,h))#,interpolation=cv2.INTER_CUBIC)

        merged1 = self.mix(fg_img=img1[0], mask_fg=mask1[0], bg_img=bg_img_new1, do_smoothing=True, do_erosion=True)
        merged2 = self.mix(fg_img=img2[0], mask_fg=mask2[0],  bg_img=bg_img_new2, do_smoothing=True, do_erosion=True)
        
        # merged1=self.transform(merged1)
        # merged2=self.transform(merged2)
        sample["transformed_image1"]=self.transform(merged1).unsqueeze(0)
        sample["transformed_image2"]=self.transform(merged2).unsqueeze(0)

        return sample

    def __iter__(self):
        batches = iter(self.dl)
        rid1 = random.randint(0, 521)
        rid2 = random.randint(0, 521)
    # bg_image_new_path = os.path.join(base_path, 'background_subtraction/background_examples/bg_new/%05d.jpg' % rid)
        bg_image_new_path1 = os.path.join(BG_PIC_PATH, BG_INDS[rid1])
        bg_img_new1 = cv2.cvtColor(cv2.imread(bg_image_new_path1),cv2.COLOR_BGR2RGB)

        bg_image_new_path2 = os.path.join(BG_PIC_PATH, BG_INDS[rid2])
        bg_img_new2 = cv2.cvtColor(cv2.imread(bg_image_new_path2),cv2.COLOR_BGR2RGB)
        for b in batches:
            yield (self.change_batch_background(b,bg_img_new1,bg_img_new2))

# train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
# train_dl = WrappedDataLoader(train_dl, preprocess)
# valid_dl = WrappedDataLoader(valid_dl, preprocess)