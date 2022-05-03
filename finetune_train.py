'''  Fine tune the trained model on a gesture classification task
Ref:https://github.com/AruniRC/resnet-face-pytorch/blob/master/finetune.py

Modify the above training pipeline to accommodate the senz3d Gesture 
recognition dataset

'''
#%%
from curses import keyname
import datetime
import math
import os
import os.path as osp
import shutil

import numpy as np
import PIL.Image
import pytz
import scipy.misc
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

# import utils
# import gc
from torchvision import transforms
from torchvision.datasets import DatasetFolder,ImageFolder
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset

from src.finetune.prepare_senz3dData import senz3d_train,senz3d_val
## Prepare DataSet: Senz3d Gesture
# data_root='/home/zlc/cll/data/senz3d_dataset_cp/acquisitions/'
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.notebook import tqdm
from src.finetune.my_wrap import Wrap_Resnet
# from tensorboard.summary import 
use_pretrain=True
save_base='/home/zlc/cll/code/peclr_cbg/data/models/finetune/hybrid2_frei_bs256_ep286_r50/'
save_base='/home/zlc/cll/code/peclr_cbg/data/models/finetune/imgnet_r50/'
save_base='/home/zlc/cll/code/peclr_cbg/data/models/finetune/hybrid2_frei_cgb_ep296_r50/'
# save_base = ''
path_pretrained_res='/home/zlc/cll/code/peclr_cbg/data/models/finetune/hybrid2_frei_bs256_ep286_r50.pth'

path_pretrained_res='/home/zlc/cll/code/peclr_cbg/data/models/finetune/frei_pretrained_cgb_ep296_res50.pth'
lr=0.001
device='cuda:1'
epochs=3
batch_size=64
os.mkdir(save_base)
#%%
train_loader = torch.utils.data.DataLoader(
    senz3d_train,
    batch_size=batch_size,
    #num_workers=4,
    shuffle=True,
    #pin_memory=True
)
valid_loader = torch.utils.data.DataLoader(
    senz3d_val,
    batch_size=batch_size,
    #num_workers=4,
    shuffle=True,
    #pin_memory=True
)
#%%
def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


criterion = nn.CrossEntropyLoss()

# output = Variable(torch.randn(10, 5).float())
# target = Variable(torch.FloatTensor(10).uniform_(0, 5).long())

# loss = criterion(output, target)
# optimizer

#%% Prepare Model
num_classes=senz3d_train.class_num()
if use_pretrain:
    model=Wrap_Resnet(num_classes,path_pretrained_res)
else:
    model=Wrap_Resnet(num_classes)
model=model.to(device)

#%%

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
# scheduler



Loss_val_list = []
Accuracy_val_list = []
Loss_list = []
Accuracy_list = []
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    Loss_list.append( epoch_loss)
    Accuracy_list.append( epoch_accuracy)
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)
            
    Loss_val_list.append( epoch_val_loss)
    Accuracy_val_list.append( epoch_val_accuracy)
    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
    if (epoch%20==1):
        torch.save(model.state_dict(), os.path.join(save_base,f'Para_hybrid2_frei_bs256_ep286_r50_ep{epoch}.pth') )

#%%
#save model
# save_base='/home/zlc/cll/code/peclr_cbg/data/models/finetune/hybrid2_frei_bs256_ep286_r50/'

torch.save(model, os.path.join(save_base,'Whole_hybrid2_frei_bs256_ep286_r50.pth') )


# 保存网络中的参数, 速度快，占空间少
torch.save(model.state_dict(),os.path.join(save_base,'Para_hybrid2_frei_bs256_ep286_r50.pth') )
np.save(save_base+'Accuracy_list.npy',Accuracy_list)
np.save(save_base+'Accuracy__val_list.npy',Accuracy_val_list)
np.save(save_base+'Loss_val_list.npy',Loss_val_list)
np.save(save_base+'Loss_list.npy',Loss_list)


# scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

#         # Saving the best performing model
#         is_best = val_acc > self.best_acc
#         if is_best:
#             self.best_acc = val_acc

#         torch.save({
#             'epoch': self.epoch,
#             'iteration': self.iteration,
#             'arch': self.model.__class__.__name__,
#             'optim_state_dict': self.optim.state_dict(),
#             'model_state_dict': self.model.state_dict(),
#             'best_acc': self.best_acc,
#         }, osp.join(self.out, 'checkpoint.pth.tar'))

#         if is_best:
#             shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
#                         osp.join(self.out, 'model_best.pth.tar'))

#         if training:
#             self.model.module.fc.train()


#     # -----------------------------------------------------------------------------
#     def train(self):
#     # -----------------------------------------------------------------------------
#         max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
#         print ('Number of iters in an epoch: %d' % len(self.train_loader))
#         print ('Total epochs: %d' % max_epoch )       

#         for epoch in tqdm.trange(self.epoch, max_epoch,
#                                  desc='Train epochs', ncols=80, leave=True):
#             self.epoch = epoch

#             if self.lr_decay_epoch is None:
#                 pass
#             else:
#                 assert self.lr_decay_epoch < max_epoch
#                 lr_scheduler(self.optim, self.epoch, 
#                              init_lr=self.init_lr, 
#                              lr_decay_epoch=self.lr_decay_epoch)

#             self.train_epoch()
#             if self.iteration >= self.max_iter:
#                 break

# %%
import matplotlib.pyplot as plt
import numpy as np
x = range(epochs)
Accuracy_list=np.load(save_base+'Accuracy_list.npy')
Accuracy_val_list=np.load(save_base+'Accuracy_val_list.npy')
Loss_list=np.load(save_base+'Loss_list.npy')
Loss_val_list=np.load(save_base+'Loss_val_list.npy')


fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(220+1)
ax.plot( Accuracy_list,label='train_acc')
ax.plot(Accuracy_val_list, label = 'val_acc')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
#plt.ylim([0.5, 1])
ax.set_title('Accuracy vs. epoches')
ax.legend(loc='lower right')

ax=fig.add_subplot(220+1)
plt.subplot(2, 1, 1)
plt.plot( Accuracy_list,label='train_acc')
plt.plot(Accuracy_val_list, label = 'val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

y_major_locator=plt.MultipleLocator(0.1)
#把y轴刻度间隔0.1
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)

plt.ylim([0, 1])
plt.title('Accuracy vs. epoches')
plt.legend(loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(Loss_list,label='train_loss')
plt.plot(Loss_val_list,label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. epoches')
plt.legend(loc='upper right')

fig.tight_layout()

plt.savefig('AccLoss_ViTTangoMulti.png',bbox_inches='tight')
plt.show()