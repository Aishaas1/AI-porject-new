"""Two predeclared preprocessing trials; selection uses validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from model import GoldCNN, CLASSES, image_tensor
from data_engineering import engineered_tensor, augment_batch


def load_images(root, config, exclusions=()):
    images, labels = [], []
    for label,name in enumerate(CLASSES):
        class_start=len(images)
        paths=sorted(p for p in (Path(root)/name).glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
        if not paths:
            raise ValueError(f'No images in {Path(root)/name}')
        for path in paths:
            if f'{Path(root).name}/{name}/{path.name}' in exclusions:
                continue
            with Image.open(path) as image:
                images.append(image_tensor(image,config))
            labels.append(label)
        if len(images)==class_start:
            raise ValueError(f'No retained images in {Path(root)/name}')
    return TensorDataset(torch.stack(images),torch.tensor(labels))


def validate_membership(data, exclusions):
    root=Path(data)
    for relative in exclusions:
        path=Path(relative)
        if len(path.parts)!=3 or path.parts[0]!='train' or path.parts[1] not in CLASSES:
            raise ValueError('Exclusions must name training images only')
        if not (root/path).is_file():
            raise ValueError(f'Excluded image is missing: {relative}')
    hashes={}
    for split in ('train','val'):
        hashes[split]=set()
        for label in CLASSES:
            for path in (root/split/label).glob('*'):
                if path.suffix.lower() in {'.jpg','.jpeg','.png','.webp'} and str(path.relative_to(root)) not in exclusions:
                    hashes[split].add(hashlib.sha256(path.read_bytes()).hexdigest())
    if hashes['train'] & hashes['val']:
        raise ValueError('Duplicate images across training and validation')


def fit_trial(data,output,config,epochs=30,augment=True,exclusions=()):
    if epochs<1:
        raise ValueError('epochs must be positive')
    validate_membership(data,exclusions)
    torch.manual_seed(42)
    model=GoldCNN()
    training=load_images(Path(data)/'train',config,exclusions)
    validation=load_images(Path(data)/'val',config)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=3)
    counts=torch.bincount(training.tensors[1],minlength=2).float()
    criterion=torch.nn.CrossEntropyLoss(weight=len(training)/(2*counts))
    best=-1.
    history=[]
    stale=0
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(epochs):
        model.train()
        loss_sum=0.
        for images,labels in DataLoader(training,batch_size=32,shuffle=True):
            if augment:
                images=augment_batch(images)
            optimizer.zero_grad()
            loss=criterion(model(images),labels)
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()*len(labels)
        model.eval()
        matrix=torch.zeros((2,2),dtype=torch.int64)
        with torch.inference_mode():
            for images,labels in DataLoader(validation,batch_size=32):
                for actual,predicted in zip(labels,model(images).argmax(1)):
                    matrix[actual,predicted]+=1
        accuracy=matrix.diag().sum().item()/len(validation)
        record=dict(epoch=epoch+1,loss=loss_sum/len(training),validation_accuracy=accuracy,
                    confusion_matrix=matrix.tolist(),learning_rate=optimizer.param_groups[0]['lr'])
        history.append(record)
        print(json.dumps(dict(trial=output.stem,**record)),flush=True)
        if accuracy>best:
            best=accuracy
            stale=0
            metrics=dict(**record,classes=CLASSES,train_images=len(training),val_images=len(validation))
            torch.save(dict(state_dict=model.state_dict(),classes=CLASSES,preprocessing=config,
                            exclusions=list(exclusions),augmentation=augment,metrics=metrics),output)
            output.with_suffix('.json').write_text(json.dumps(metrics,indent=2))
        else:
            stale+=1
        scheduler.step(accuracy)
        output.with_name(output.stem+'_history.json').write_text(json.dumps(history,indent=2))
        if stale>=7:
            break
    return dict(path=str(output),validation_accuracy=best,preprocessing=config,augmentation=augment)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/goldnet')
    p.add_argument('--exclusions',default='reports/engineering_exclusions.json')
    p.add_argument('--epochs',type=int,default=30)
    args=p.parse_args()
    if args.epochs<1:
        p.error('epochs must be positive')
    results=[]
    exclusions=json.loads(Path(args.exclusions).read_text())['exclude_from_training']
    trials=[('clean_control',None,False),
            ('reflect_aug',dict(version='reflect_v1',size=128,luminance_autocontrast=False),True),
            ('luminance_aug',dict(version='reflect_v1',size=128,luminance_autocontrast=True),True)]
    for name,config,augment in trials:
        results.append(fit_trial(args.data,f'artifacts/{name}.pt',config,args.epochs,augment,exclusions))
    Path('reports/engineering_trials.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(dict(best=max(results,key=lambda r:r['validation_accuracy']))),flush=True)
