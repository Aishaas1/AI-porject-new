"""ImageNet transfer learning with official torchvision ResNet18 weights."""
import os
from pathlib import Path
import torch
from torch import nn
from PIL import ImageOps
import numpy as np
os.environ.setdefault('TORCH_HOME',str(Path(__file__).resolve().parent.parent/'torch-cache'))

CONFIG = {'version':'imagenet_v1','size':224}


def transfer_tensor(image):
    image=ImageOps.pad(ImageOps.exif_transpose(image).convert('RGB'),(224,224))
    tensor=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()/255).permute(2,0,1)
    return (tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


class GoldResNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import resnet18,ResNet18_Weights
        self.backbone=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc=nn.Identity()
        self.head=nn.Sequential(nn.Dropout(.25),nn.Linear(512,2))
        self.preprocessing=CONFIG.copy()
        self.architecture='ResNet18'
    def forward(self,x):
        return self.head(self.backbone(x))
    def freeze(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad=False
    def fine_tune(self, blocks=1):
        if blocks not in (1, 2):
            raise ValueError('Fine tuning supports one or two final blocks')
        self.freeze()
        layers = [self.backbone.layer4]
        if blocks == 2:
            layers.insert(0, self.backbone.layer3)
        for layer in layers:
            for parameter in layer.parameters():
                parameter.requires_grad=True
        for module in self.backbone.modules():
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad=False
    def train(self,mode=True):
        super().train(mode)
        # Never update pretrained BN statistics, including during fine tuning.
        for module in self.backbone.modules():
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
        return self
