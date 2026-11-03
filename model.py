import numpy as np
import torch
from PIL import ImageOps
from torch import nn

CLASSES = ['fake', 'real']

class GoldCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Dropout(.3), nn.Linear(1024, 2))

    def forward(self, x):
        return self.layers(x)

def image_tensor(image, preprocessing=None):
    if preprocessing and preprocessing.get('version') == 'imagenet_v1':
        from transfer_model import transfer_tensor
        return transfer_tensor(image)
    if preprocessing:
        from data_engineering import engineered_tensor
        return engineered_tensor(image, preprocessing)
    image = ImageOps.pad(ImageOps.exif_transpose(image).convert('RGB'), (128, 128))
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255).permute(2, 0, 1)

def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('classes') != CLASSES:
        raise ValueError('Incompatible classes')
    architecture = checkpoint.get('architecture', 'goldcnn')
    if architecture == 'resnet18':
        from transfer_model import GoldResNet
        model = GoldResNet(pretrained=False)
    elif architecture == 'goldcnn':
        model = GoldCNN()
    else:
        raise ValueError('Unsupported model architecture')
    model.load_state_dict(checkpoint['state_dict'])
    model.preprocessing = checkpoint.get('preprocessing')
    return model.eval()
