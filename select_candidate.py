"""Validation-only selection of standalone/fused CNNs and decision thresholds."""
import hashlib,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model import load_model,CLASSES
from efficient_model import GoldFusion,GoldTTA,CONFIG
from train_engineered import load_images


def probabilities(model,dataset,flip=False,resize=None):
    model.eval(); chunks=[]
    with torch.inference_mode():
        for x,_ in DataLoader(dataset,batch_size=16):
            if flip:x=x.flip(-1)
            if resize is not None:
                x=torch.nn.functional.interpolate(x,size=(resize,resize),mode='bilinear',align_corners=False,antialias=True)
            chunks.append(model(x).softmax(1))
    return torch.cat(chunks)


def select(data='data/goldnet',output='artifacts/gold_selected.pt'):
    resnet=load_model('artifacts/gold_finetuned.pt')
    efficient_path=Path('artifacts/gold_efficient_adapted.pt')
    if efficient_path.is_file():
        metadata=torch.load(efficient_path,map_location='cpu',weights_only=True)
        if metadata['adaptation_initial_sha256']!=hashlib.sha256(Path('artifacts/gold_efficient.pt').read_bytes()).hexdigest():
            raise ValueError('Stale BN adaptation; rerun adapt_bn.py')
    else:
        efficient_path=Path('artifacts/gold_efficient.pt')
    efficient=load_model(efficient_path)
    val224=load_images(Path(data)/'val',resnet.preprocessing)
    val320=load_images(Path(data)/'val',CONFIG)
    labels=val224.tensors[1]
    assert torch.equal(labels,val320.tensors[1])
    fusion=GoldFusion()
    fusion.resnet.load_state_dict(resnet.state_dict())
    fusion.efficient.load_state_dict(efficient.state_dict())
    candidates=[]
    candidates.append(('resnet',probabilities(resnet,val224),probabilities(resnet,val224,True),resnet.preprocessing))
    ep,ef=probabilities(efficient,val320),probabilities(efficient,val320,True)
    candidates.append(('efficient',ep,ef,CONFIG))
    rp,rf=probabilities(resnet,val320,resize=224),probabilities(resnet,val320,True,resize=224)
    # Cache component probabilities, reproducing the production resize/normalization.

    for weight in (.25,.5,.75):
        fusion.weight=weight
        p=((1-weight)*rp+weight*ep).clamp_min(1e-8).log().softmax(1)
        flipped=((1-weight)*rf+weight*ef).clamp_min(1e-8).log().softmax(1)
        candidates.append((f'fusion_{weight}',p,flipped,CONFIG))
    trials=[];best=None;best_score=None
    for name,probs,flipped,config in candidates:
        for use_tta in (False,True):
            scores=((probs+flipped)/2).clamp_min(1e-8).log().softmax(1) if use_tta else probs
            for threshold in (.4,.45,.5,.55,.6):
                prediction=(scores[:,1]>threshold).long()
                matrix=torch.zeros((2,2),dtype=torch.int64)
                for a,p in zip(labels,prediction):matrix[a,p]+=1
                score=(int(matrix.diag().sum()),-int(matrix[0,1]),-abs(threshold-.5),-int(use_tta))
                record=dict(candidate=name,flip_tta=use_tta,threshold=threshold,validation_accuracy=int(matrix.diag().sum())/len(labels),confusion_matrix=matrix.tolist())
                trials.append(record)
                if best_score is None or score>best_score:
                    best_score=score;best=(record,config)
    record,config=best
    name=record['candidate']
    if name=='resnet':
        checkpoint=torch.load('artifacts/gold_finetuned.pt',map_location='cpu',weights_only=True)
    elif name=='efficient':
        checkpoint=torch.load(efficient_path,map_location='cpu',weights_only=True)
    else:
        weight=float(name.split('_')[1]);fusion.weight=weight
        checkpoint=dict(architecture='fusion',fusion_weight=weight,state_dict=fusion.state_dict(),classes=CLASSES,preprocessing=config)
    if record['flip_tta']:
        if name=='resnet':base=resnet
        elif name=='efficient':base=efficient
        else:base=fusion
        wrapper=GoldTTA(base).eval()
        checkpoint={**checkpoint,'base_architecture':checkpoint['architecture'],
                    'architecture':'tta','state_dict':wrapper.state_dict()}
    checkpoint['decision_threshold']=record['threshold']
    checkpoint['selection_metrics']=record
    torch.save(checkpoint,output)
    # Verify the exported artifact, not only cached candidate arithmetic.
    exported=load_model(output)
    dataset=val224 if config['size']==224 else val320
    exported_probs=probabilities(exported,dataset)
    predicted=(exported_probs[:,1]>record['threshold']).long()
    check=torch.zeros((2,2),dtype=torch.int64)
    for a,p in zip(labels,predicted):check[a,p]+=1
    if check.tolist()!=record['confusion_matrix']:
        raise ValueError('Exported model does not reproduce validation selection')
    report=dict(selected=record,trials=trials,selection_split='validation_only',test_used=False,
                checkpoint_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest(),
                note='50 predeclared model/threshold/TTA combinations; scores are not calibrated. Training membership and validation/test membership unchanged.')
    Path('reports/candidate_selection.json').write_text(json.dumps(report,indent=2))
    Path(output).with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(record),flush=True)
    return report

if __name__=='__main__':select()
