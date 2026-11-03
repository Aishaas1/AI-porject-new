"""Shared deterministic image preprocessing and training-only augmentation."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageOps


def engineered_tensor(image, config):
    if config.get('version') != 'reflect_v1':
        raise ValueError('Unsupported preprocessing version')
    size = int(config.get('size',128))
    if size < 32 or size > 512:
        raise ValueError('Unsupported image size')
    image = ImageOps.exif_transpose(image).convert('RGB')
    if config.get('luminance_autocontrast',False):
        # Adjust luminance only; do not independently equalize RGB/hue channels.
        y, cb, cr = image.convert('YCbCr').split()
        y = ImageOps.autocontrast(y,cutoff=1)
        image = Image.merge('YCbCr',(y,cb,cr)).convert('RGB')
    ratio = size/max(image.size)
    width, height = max(1,round(image.width*ratio)), max(1,round(image.height*ratio))
    array = np.asarray(image.resize((width,height),Image.Resampling.LANCZOS))
    top, left = (size-height)//2, (size-width)//2
    mode = 'reflect' if min(width,height)>1 else 'edge'
    array = np.pad(array,((top,size-height-top),(left,size-width-left),(0,0)),mode=mode)
    return torch.from_numpy(array.astype(np.float32)/255).permute(2,0,1)


def augment_batch(images, generator=None):
    """Mild geometry/exposure changes; evaluation must never call this function."""
    batch = images.shape[0]
    def random(*shape):
        return torch.rand(shape,generator=generator,device=images.device)
    angles = (random(batch)*2-1)*(12*math.pi/180)
    scale = .95+.10*random(batch)
    flip = torch.where(random(batch)<.5,-1.,1.)
    theta = torch.zeros((batch,2,3),device=images.device,dtype=images.dtype)
    theta[:,0,0] = torch.cos(angles)*scale*flip
    theta[:,0,1] = -torch.sin(angles)*scale
    theta[:,1,0] = torch.sin(angles)*scale*flip
    theta[:,1,1] = torch.cos(angles)*scale
    theta[:,:,2] = (random(batch,2)*2-1)*.04
    grid = F.affine_grid(theta,images.shape,align_corners=False)
    result = F.grid_sample(images,grid,padding_mode='reflection',align_corners=False)
    brightness = .9+.2*random(batch,1,1,1)
    contrast = .9+.2*random(batch,1,1,1)
    mean = result.mean(dim=(2,3),keepdim=True)
    return ((result-mean)*contrast+mean).mul(brightness).clamp(0,1)
