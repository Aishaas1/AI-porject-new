import io
import torch
import pytest
from PIL import Image
from torch import nn
from model import load_model,image_tensor
from efficient_model import GoldEfficientNet,GoldFusion,CONFIG
from app import create_app


def test_efficient_freezing_and_threshold_serving(tmp_path):
    torch.set_num_threads(2)
    model=GoldEfficientNet();model.fine_tune();model.train()
    assert not any(p.requires_grad for p in model.backbone.features[0].parameters())
    assert any(p.requires_grad for p in model.backbone.features[-1].parameters())
    assert all(not m.training and not any(p.requires_grad for p in m.parameters())
               for m in model.modules() if isinstance(m,nn.BatchNorm2d))
    model.eval()
    image=Image.new('RGB',(80,120),(200,150,70))
    x=image_tensor(image,CONFIG)[None]
    assert x.shape==(1,3,320,320)
    with torch.inference_mode(): expected=model(x).softmax(1)[0]
    threshold=.6
    path=tmp_path/'efficient.pt'
    torch.save(dict(architecture='efficientnet_b0',state_dict=model.state_dict(),classes=['fake','real'],preprocessing=CONFIG,decision_threshold=threshold),path)
    restored=load_model(path)
    with torch.inference_mode():torch.testing.assert_close(restored(x),model(x))
    b=io.BytesIO();image.save(b,format='PNG')
    response=create_app(path).test_client().post('/predict',data={'image':(io.BytesIO(b.getvalue()),'sample.png')})
    assert response.status_code==200
    assert response.json['decision_threshold']==threshold
    assert response.json['label']==('حقيقي' if expected[1]>.6 else 'مزيف')
    assert response.json['real_score']==round(expected[1].item()*100,2)


def test_fusion_probability_and_serialization(tmp_path):
    torch.set_num_threads(2)
    model=GoldFusion(.25).eval()
    x=torch.randn(1,3,320,320)
    with torch.inference_mode():
        actual=model(x).softmax(1)
        small=torch.nn.functional.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
        expected=.75*model.resnet(small).softmax(1)+.25*model.efficient(x).softmax(1)
    torch.testing.assert_close(actual,expected)
    path=tmp_path/'fusion.pt'
    torch.save(dict(architecture='fusion',fusion_weight=.25,state_dict=model.state_dict(),classes=['fake','real'],preprocessing=CONFIG),path)
    restored=load_model(path)
    with torch.inference_mode():torch.testing.assert_close(restored(x).softmax(1),actual)
    assert restored.decision_threshold==.5
    with pytest.raises(ValueError):image_tensor(Image.new('RGB',(10,10)),dict(CONFIG,size=13))


def test_bn_calibration_uses_no_grad_and_preserves_parameters():
    from adapt_bn import calibrate
    from torch.utils.data import TensorDataset
    model=nn.Sequential(nn.BatchNorm2d(3),nn.Flatten(),nn.Linear(12,2)).eval()
    original={n:p.clone() for n,p in model.named_parameters()}
    data=TensorDataset(torch.full((4,3,2,2),5.),torch.tensor([0,1,0,1]))
    calibrate(model,data)
    torch.testing.assert_close(model[0].running_mean,torch.full((3,),5.))
    assert all(torch.equal(p,original[n]) for n,p in model.named_parameters())
    assert all(p.grad is None for p in model.parameters())
    assert model[0].training is False


def test_tta_round_trip(tmp_path):
    from efficient_model import GoldTTA
    torch.set_num_threads(2)
    model=GoldTTA(GoldEfficientNet()).eval()
    x=torch.randn(1,3,320,320)
    with torch.inference_mode():expected=model(x)
    path=tmp_path/'tta.pt'
    torch.save(dict(architecture='tta',base_architecture='efficientnet_b0',classes=['fake','real'],state_dict=model.state_dict(),preprocessing=CONFIG),path)
    restored=load_model(path)
    with torch.inference_mode():torch.testing.assert_close(restored(x),expected)
