import torch
import torch.nn as nn
from torchvision import models



class Wrap_Resnet(nn.Module):
    def __init__(self, model_path,num_classes,backend_model="rn50", ):
        super().__init__()
        # Initialize a torchvision resnet
        if backend_model == "rn50":
            model_func = models.resnet50
        elif backend_model == "rn152":
            model_func = models.resnet152
        else:
            raise Exception(f"Unknown backend_model: {backend_model}")
        # backend_model = model_func()
        backend_model = torch.load(model_path)
        # backend_model.load_state_dict(checkpoint["state_dict"])
        num_feat = backend_model.fc.in_features
        # 2D + zrel for 21 keypoints: 3 * 21. Please ignore +1, it is no longer used
        backend_model.fc = nn.Linear(num_feat,num_classes)
        # Initialize the zroot refinement module
        # zroot_ref = ZrootMLP_ref()

        self.backend_model = backend_model
        # self.zroot_ref = zroot_ref
        for p in self.backend_model.parameters():
            p.requires_grad=False
        for p in self.backend_model.fc.parameters():
            p.requires_grad=True

    def forward(self, img, K=None):

        output = self.backend_model(img)

        return output
