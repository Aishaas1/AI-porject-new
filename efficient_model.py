"""EfficientNet-B0 and probability fusion; no downloads at inference."""
import torch
from torch import nn
from transfer_model import GoldResNet

CONFIG={'version':'imagenet_v1','size':320}

class GoldEfficientNet(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
        self.backbone=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        self.backbone.classifier=nn.Identity()
        self.head=nn.Sequential(nn.Dropout(.3),nn.Linear(1280,2))
        self.preprocessing=CONFIG.copy()
        self.architecture='EfficientNet-B0'
    def forward(self,x):
        return self.head(self.backbone(x))
    def freeze(self):
        for p in self.backbone.parameters(): p.requires_grad=False
    def fine_tune(self):
        self.freeze()
        for block in self.backbone.features[3:]:
            for p in block.parameters(): p.requires_grad=True
        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d):
                for p in m.parameters(): p.requires_grad=False
    def train(self,mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d): m.eval()
        return self

class GoldFusion(nn.Module):
    def __init__(self,weight=.5):
        super().__init__()
        self.resnet=GoldResNet()
        self.efficient=GoldEfficientNet()
        self.weight=weight
        self.preprocessing=CONFIG.copy()
        self.architecture='ResNet18 + EfficientNet-B0'
    def forward(self,x):
        # Input normalized at 320px. Resize the normalized tensor for ResNet.
        small=torch.nn.functional.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
        p=(1-self.weight)*self.resnet(small).softmax(1)+self.weight*self.efficient(x).softmax(1)
        return p.clamp_min(1e-8).log()

class GoldTTA(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base
        self.preprocessing=base.preprocessing
        self.architecture=base.architecture+' (flip TTA)'
    def forward(self,x):
        p=(self.base(x).softmax(1)+self.base(x.flip(-1)).softmax(1))/2
        return p.clamp_min(1e-8).log()
