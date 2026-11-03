"""Training-only BN recalibration, validated against the unchanged checkpoint."""
import copy,json,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import load_model
from efficient_model import CONFIG
from train_engineered import load_images,validate_membership
from train_transfer import matrix_for


def calibrate(model,training):
    model.eval()
    layers=[m for m in model.modules() if isinstance(m,nn.BatchNorm2d)]
    for m in layers:
        m.reset_running_stats();m.momentum=None;m.train(True)
    with torch.no_grad():
        for x,_ in DataLoader(training,batch_size=16): model(x)
    model.eval()
    return model


def adapt(data='data/goldnet'):
    initial=Path('artifacts/gold_efficient.pt')
    checkpoint=torch.load(initial,map_location='cpu',weights_only=True)
    digest=hashlib.sha256(initial.read_bytes()).hexdigest()
    model=load_model(initial)
    exclusions=json.loads(Path('reports/engineering_exclusions.json').read_text())['exclude_from_training']
    validate_membership(data,exclusions)
    training=load_images(Path(data)/'train',CONFIG,exclusions)
    validation=load_images(Path(data)/'val',CONFIG)
    original=copy.deepcopy(model.state_dict())
    calibrate(model,training)
    adapted=copy.deepcopy(model.state_dict())
    records=[];best=(-1,-100000);selected=None
    for alpha in (0.,.25,.5,.75,1.):
        state={key:(original[key]*(1-alpha)+value*alpha if key.endswith(('running_mean','running_var')) else original[key]) for key,value in adapted.items()}
        model.load_state_dict(state);model.eval()
        matrix=matrix_for(model,validation)
        record=dict(alpha=alpha,validation_accuracy=int(matrix.diag().sum())/len(validation),confusion_matrix=matrix.tolist())
        records.append(record);print(json.dumps(record),flush=True)
        score=(int(matrix.diag().sum()),-int(matrix[0,1]))
        if score>best:
            best=score;selected=record
            torch.save({**checkpoint,'state_dict':model.state_dict(),'bn_adaptation':record,
                        'adaptation_initial_sha256':digest},'artifacts/gold_efficient_adapted.pt')
    Path('reports/bn_adaptation.json').write_text(json.dumps(dict(training_only=True,initial_sha256=digest,selected=selected,trials=records),indent=2))
    return selected

if __name__=='__main__':adapt()
