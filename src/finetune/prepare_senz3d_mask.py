#%%
import torch
import torch.hub
import PIL
from PIL import Image
# Create the model
model = torch.hub.load(
    repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
    model='hand_segmentor', 
    pretrained=True
)
img_path='/home/zlc/cll/data/senz3d_dataset_cp/acquisitions/G1/S1_G1_1-color.png'
img=Image.Open(img_path)
#%%
# Inference
model.eval()
img_rnd = torch.randn(1, 3, 256, 256) # [B, C, H, W]
preds = model(img_rnd).argmax(1) # [B, H, W]