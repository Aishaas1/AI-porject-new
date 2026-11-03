"""Validation-selected frozen-head + low-LR fine tuning on cleaned GoldNet."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader,TensorDataset
from transfer_model import GoldResNet,CONFIG
from train_engineered import load_images,validate_membership
from model import CLASSES


def matrix_for(model,dataset):
    model.eval()
    matrix=torch.zeros((2,2),dtype=torch.int64)
    with torch.inference_mode():
        for images,labels in DataLoader(dataset,batch_size=16):
            for actual,predicted in zip(labels,model(images).argmax(1)):
                matrix[actual,predicted]+=1
    return matrix


def train(data,output,head_epochs=25,fine_epochs=6):
    torch.manual_seed(42)
    exclusions=json.loads(Path('reports/engineering_exclusions.json').read_text())['exclude_from_training']
    validate_membership(data,exclusions)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    print('Loading cleaned training/validation only; test not loaded.',flush=True)
    training=load_images(Path(data)/'train',CONFIG,exclusions)
    validation=load_images(Path(data)/'val',CONFIG)
    model=GoldResNet(pretrained=True)
    model.freeze()
    print('Extracting frozen ImageNet features once.',flush=True)
    def features(dataset):
        chunks=[]
        model.eval()
        with torch.no_grad():
            for images,_ in DataLoader(dataset,batch_size=16):
                chunks.append(model.backbone(images).detach())
        return TensorDataset(torch.cat(chunks),dataset.tensors[1])
    train_features=features(training)
    val_features=features(validation)
    best=(-1,-10**9)
    history=[]
    def consider(matrix,stage,epoch,loss):
        nonlocal best
        accuracy=matrix.diag().sum().item()/len(validation)
        record=dict(stage=stage,epoch=epoch,loss=loss,validation_accuracy=accuracy,
                    confusion_matrix=matrix.tolist(),train_images=len(training),val_images=len(validation))
        history.append(record)
        print(json.dumps(record),flush=True)
        # Accuracy first; fewer counterfeit-to-real errors break exact ties.
        score=(int(matrix.diag().sum()),-int(matrix[0,1]))
        if score>best:
            best=score
            torch.save(dict(architecture='resnet18',state_dict=model.state_dict(),classes=CLASSES,
                            preprocessing=CONFIG,metrics=record,exclusions=exclusions,
                            source_weights='torchvision ResNet18_Weights.IMAGENET1K_V1'),output)
            output.with_suffix('.json').write_text(json.dumps(record,indent=2))
        output.with_name(output.stem+'_history.json').write_text(json.dumps(history,indent=2))
    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-3,weight_decay=1e-3)
    for epoch in range(head_epochs):
        model.head.train()
        total=0.
        for vectors,labels in DataLoader(train_features,batch_size=32,shuffle=True):
            optimizer.zero_grad()
            loss=loss_fn(model.head(vectors),labels)
            loss.backward()
            optimizer.step()
            total+=loss.item()*len(labels)
        matrix=matrix_for(model.head,val_features)
        consider(matrix,'frozen_head',epoch+1,total/len(training))
    model.load_state_dict(torch.load(output,map_location='cpu',weights_only=True)['state_dict'])
    model.fine_tune()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-4)
    for epoch in range(fine_epochs):
        model.train()
        total=0.
        for images,labels in DataLoader(training,batch_size=16,shuffle=True):
            optimizer.zero_grad()
            loss=loss_fn(model(images),labels)
            loss.backward()
            optimizer.step()
            total+=loss.item()*len(labels)
        consider(matrix_for(model,validation),'fine_tune',epoch+1,total/len(training))
    return history

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/goldnet')
    p.add_argument('--output',default='artifacts/gold_transfer.pt')
    p.add_argument('--head-epochs',type=int,default=25)
    p.add_argument('--fine-epochs',type=int,default=6)
    args=p.parse_args()
    if min(args.head_epochs,args.fine_epochs)<1:
        p.error('Epoch counts must be positive')
    train(args.data,args.output,args.head_epochs,args.fine_epochs)
