"""Fixed clean-split experiment: EfficientNet at 320px, no test-set loading."""
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader,TensorDataset
from efficient_model import GoldEfficientNet,CONFIG
from model import CLASSES
from train_engineered import load_images,validate_membership
from train_transfer import matrix_for


def train(data,output,head_epochs=35,fine_epochs=12):
    torch.manual_seed(42)
    exclusions=json.loads(Path('reports/engineering_exclusions.json').read_text())['exclude_from_training']
    validate_membership(data,exclusions)
    print('Loading cleaned train and validation at 320px. Test is not loaded.',flush=True)
    training=load_images(Path(data)/'train',CONFIG,exclusions)
    validation=load_images(Path(data)/'val',CONFIG)
    model=GoldEfficientNet(pretrained=True)
    model.freeze(); model.eval()
    def features(dataset):
        result=[]
        with torch.no_grad():
            for x,_ in DataLoader(dataset,batch_size=16): result.append(model.backbone(x))
        return TensorDataset(torch.cat(result),dataset.tensors[1])
    print('Extracting frozen ImageNet features.',flush=True)
    tr,va=features(training),features(validation)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    history=[]; best=(-1,-100000)
    def consider(matrix,stage,epoch,loss):
        nonlocal best
        record=dict(stage=stage,epoch=epoch,loss=loss,validation_accuracy=int(matrix.diag().sum())/len(validation),confusion_matrix=matrix.tolist(),train_images=len(training),val_images=len(validation))
        history.append(record); print(json.dumps(record),flush=True)
        score=(int(matrix.diag().sum()),-int(matrix[0,1]))
        improved=score>best
        if improved:
            best=score
            torch.save(dict(architecture='efficientnet_b0',state_dict=model.state_dict(),classes=CLASSES,preprocessing=CONFIG,metrics=record,exclusions=exclusions,source_weights='EfficientNet_B0_Weights.IMAGENET1K_V1'),output)
            output.with_suffix('.json').write_text(json.dumps(record,indent=2))
        output.with_name(output.stem+'_history.json').write_text(json.dumps(history,indent=2))
        return improved
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-3,weight_decay=1e-3)
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=.05)
    for epoch in range(head_epochs):
        model.head.train(); total=0.
        for x,y in DataLoader(tr,batch_size=32,shuffle=True):
            optimizer.zero_grad(); loss=criterion(model.head(x),y); loss.backward(); optimizer.step();total+=loss.item()*len(y)
        consider(matrix_for(model.head,va),'head',epoch+1,total/len(tr))
    model.load_state_dict(torch.load(output,map_location='cpu',weights_only=True)['state_dict'])
    model.fine_tune()
    optimizer=torch.optim.AdamW([
        {'params':[p for b in model.backbone.features[3:6] for p in b.parameters() if p.requires_grad],'lr':1e-5},
        {'params':[p for b in model.backbone.features[6:] for p in b.parameters() if p.requires_grad],'lr':3e-5},
        {'params':model.head.parameters(),'lr':1e-4}],weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=fine_epochs,eta_min=1e-6)
    stale=0
    for epoch in range(fine_epochs):
        model.train(); total=0.
        for x,y in DataLoader(training,batch_size=16,shuffle=True):
            x=x.clone(); mask=torch.rand(len(x))<.5; x[mask]=x[mask].flip(-1)
            optimizer.zero_grad();loss=criterion(model(x),y);loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step();total+=loss.item()*len(y)
        improved=consider(matrix_for(model,validation),'fine_tune',epoch+1,total/len(training))
        stale=0 if improved else stale+1
        scheduler.step()
        if stale>=5:break
    return history

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/goldnet');p.add_argument('--output',default='artifacts/gold_efficient.pt');p.add_argument('--head-epochs',type=int,default=35);p.add_argument('--fine-epochs',type=int,default=12)
    args=p.parse_args()
    if min(args.head_epochs,args.fine_epochs)<1:p.error('Epoch counts must be positive')
    train(args.data,args.output,args.head_epochs,args.fine_epochs)
