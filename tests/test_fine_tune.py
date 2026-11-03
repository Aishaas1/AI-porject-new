import hashlib
import json
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from transfer_model import GoldResNet,CONFIG
from fine_tune import fine_tune


def test_two_blocks_and_bn_frozen():
    model=GoldResNet()
    model.fine_tune(blocks=2)
    model.train()
    assert not any(p.requires_grad for p in model.backbone.layer2.parameters())
    assert any(p.requires_grad for p in model.backbone.layer3.parameters())
    assert any(p.requires_grad for p in model.backbone.layer4.parameters())
    assert all(not m.training and not any(p.requires_grad for p in m.parameters())
               for m in model.modules() if isinstance(m,nn.BatchNorm2d))
    with pytest.raises(ValueError):
        model.fine_tune(blocks=3)


def test_fine_tune_preserves_initial_and_records_baseline(tmp_path):
    torch.set_num_threads(2)
    model=GoldResNet()
    initial=tmp_path/'initial.pt'
    torch.save(dict(architecture='resnet18',classes=['fake','real'],preprocessing=CONFIG,
                    state_dict=model.state_dict()),initial)
    digest=hashlib.sha256(initial.read_bytes()).hexdigest()
    rng=np.random.default_rng(42)
    for split in ('train','val'):
        for label in ('fake','real'):
            folder=tmp_path/'data'/split/label
            folder.mkdir(parents=True)
            Image.fromarray(rng.integers(0,256,(30,45,3),dtype=np.uint8)).save(folder/'sample.png')
    exclusions=tmp_path/'exclude.json'
    exclusions.write_text(json.dumps({'exclude_from_training':[]}))
    output=tmp_path/'fine.pt'
    record=fine_tune(tmp_path/'data',initial,output,epochs=1,exclusions_path=exclusions)
    assert hashlib.sha256(initial.read_bytes()).hexdigest()==digest
    assert record['epoch'] in (0,1)
    assert record['val_images']==2
    checkpoint=torch.load(output,map_location='cpu',weights_only=True)
    assert checkpoint['fine_tuning']['initial_sha256']==digest
    assert checkpoint['fine_tuning']['batch_norm_frozen'] is True
    assert len(json.loads((tmp_path/'fine_history.json').read_text()))==1
    with pytest.raises(ValueError,match='separate output'):
        fine_tune(tmp_path/'data',initial,initial,epochs=1)
