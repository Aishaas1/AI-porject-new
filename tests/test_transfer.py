import io
import torch
from PIL import Image
from torch import nn
from transfer_model import GoldResNet,CONFIG,transfer_tensor
from model import load_model,CLASSES
from app import create_app


def test_freezing_and_serialized_inference(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model=GoldResNet(pretrained=False)
    model.freeze()
    assert not any(p.requires_grad for p in model.backbone.parameters())
    model.fine_tune()
    model.train()
    assert any(p.requires_grad for p in model.backbone.layer4.parameters())
    assert not any(p.requires_grad for p in model.backbone.layer3.parameters())
    assert all(not module.training for module in model.backbone.modules() if isinstance(module,nn.BatchNorm2d))
    assert all(not p.requires_grad for m in model.backbone.modules() if isinstance(m,nn.BatchNorm2d) for p in m.parameters())
    bn_before={name:value.clone() for name,value in model.named_buffers()}
    before=model.backbone.layer4[0].conv1.weight.detach().clone()
    image=Image.new('RGB',(110,170),(200,150,70))
    tensor=transfer_tensor(image)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    loss=nn.CrossEntropyLoss()(model(tensor[None]),torch.tensor([1]))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before,model.backbone.layer4[0].conv1.weight)
    assert all(torch.equal(value,bn_before[name]) for name,value in model.named_buffers())
    path=tmp_path/'model.pt'
    torch.save(dict(architecture='resnet18',state_dict=model.state_dict(),classes=CLASSES,preprocessing=CONFIG),path)
    loaded=load_model(path)
    with torch.inference_mode():
        expected=loaded(tensor[None]).softmax(1)[0]
    client=create_app(path).test_client()
    assert client.get('/health').json['architecture']=='ResNet18'
    buffer=io.BytesIO()
    image.save(buffer,format='PNG')
    response=client.post('/predict',data={'image':(io.BytesIO(buffer.getvalue()),'sample.png')})
    assert response.status_code==200
    assert response.json['real_score']==round(expected[1].item()*100,2)
