#%%
from torchvision import transforms
from torchvision.datasets import DatasetFolder,ImageFolder
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import torch
import cv2
# from src.finetune.senz3d_dataset import senz3dDataset
## Prepare DataSet: Senz3d Gesture
data_root='/home/zlc/cll/data/senz3d_dataset_cp/acquisitions/'

transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                        ),
                    ]
                )

samples_all=ImageFolder(data_root,transform)


class senz3dDataset(Dataset):
    def __init__(self, file_list,label_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.label_list = label_list

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    def class_num(self):
        return(len(set(self.label_list)))

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        
        img_transformed = self.transform(img)
        label = self.label_list[idx]
       

        return img_transformed, label

#%%
data_all=samples_all.imgs
X_all=[]
Y_all=[]
for n in data_all:
    x,y=n
    X_all.append(x)
    Y_all.append(y)

X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.33, stratify=Y_all,random_state=42)

senz3d_train=senz3dDataset(X_train,y_train,transform)
senz3d_val=senz3dDataset(X_test,y_test,transform)


# %%
