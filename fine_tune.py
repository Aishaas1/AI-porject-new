"""Extend local ResNet18 fine tuning; select on validation, never test data."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model import load_model
from transfer_model import CONFIG
from train_engineered import load_images, validate_membership
from train_transfer import matrix_for


def score(matrix):
    return (int(matrix.diag().sum()), -int(matrix[0,1]))


def fine_tune(data, initial, output, epochs=8, exclusions_path='reports/engineering_exclusions.json'):
    if epochs<1:
        raise ValueError('epochs must be positive')
    initial,output=Path(initial),Path(output)
    if initial.resolve()==output.resolve():
        raise ValueError('Use a separate output to preserve the initial checkpoint')
    torch.manual_seed(42)
    source=torch.load(initial,map_location='cpu',weights_only=True)
    if source.get('architecture')!='resnet18' or source.get('preprocessing')!=CONFIG:
        raise ValueError('Expected the project ResNet18 ImageNet preprocessing checkpoint')
    exclusions=json.loads(Path(exclusions_path).read_text())['exclude_from_training']
    validate_membership(data,exclusions)
    training=load_images(Path(data)/'train',CONFIG,exclusions)
    validation=load_images(Path(data)/'val',CONFIG)
    model=load_model(initial)
    baseline=matrix_for(model,validation)
    best=score(baseline)
    model.fine_tune(blocks=2)
    optimizer=torch.optim.AdamW([
        {'params':[p for p in model.backbone.layer3.parameters() if p.requires_grad], 'lr':2e-6},
        {'params':[p for p in model.backbone.layer4.parameters() if p.requires_grad], 'lr':5e-6},
        {'params':model.head.parameters(), 'lr':1e-5},
    ],weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=5e-7)
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=.05)
    output.parent.mkdir(parents=True,exist_ok=True)
    history=[]
    source_hash=hashlib.sha256(initial.read_bytes()).hexdigest()
    def save(epoch,matrix,loss,improved):
        record=dict(stage='extended_fine_tune',epoch=epoch,loss=loss,
                    validation_accuracy=int(matrix.diag().sum())/len(validation),
                    confusion_matrix=matrix.tolist(),train_images=len(training),val_images=len(validation),
                    improved_over_initial=improved)
        source_copy={**source,'state_dict':model.state_dict(),'metrics':record,
                     'fine_tuning':{'blocks':2,'batch_norm_frozen':True,'initial_sha256':source_hash,
                                    'initial_checkpoint':str(initial),'horizontal_flip':True,'label_smoothing':.05}}
        torch.save(source_copy,output)
        output.with_suffix('.json').write_text(json.dumps(record,indent=2))
    # The existing model is a candidate, so validation cannot silently regress.
    save(0,baseline,None,False)
    print(json.dumps(dict(stage='baseline',validation_accuracy=int(baseline.diag().sum())/len(validation))),flush=True)
    stale=0
    for epoch in range(epochs):
        model.train()
        total=0.
        for images,labels in DataLoader(training,batch_size=16,shuffle=True):
            images=images.clone()
            mask=torch.rand(len(images))<.5
            images[mask]=images[mask].flip(-1)
            optimizer.zero_grad()
            loss=criterion(model(images),labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step()
            total+=loss.item()*len(labels)
        matrix=matrix_for(model,validation)
        record=dict(epoch=epoch+1,loss=total/len(training),validation_accuracy=int(matrix.diag().sum())/len(validation),
                    confusion_matrix=matrix.tolist(),learning_rates=[g['lr'] for g in optimizer.param_groups])
        history.append(record)
        print(json.dumps(record),flush=True)
        if score(matrix)>best:
            best=score(matrix)
            stale=0
            save(epoch+1,matrix,record['loss'],True)
        else:
            stale+=1
        scheduler.step()
        output.with_name(output.stem+'_history.json').write_text(json.dumps(history,indent=2))
        if stale>=4:
            break
    return json.loads(output.with_suffix('.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/goldnet')
    p.add_argument('--initial',default='artifacts/gold_transfer.pt')
    p.add_argument('--output',default='artifacts/gold_finetuned.pt')
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--exclusions',default='reports/engineering_exclusions.json')
    args=p.parse_args()
    fine_tune(args.data,args.initial,args.output,args.epochs,args.exclusions)
