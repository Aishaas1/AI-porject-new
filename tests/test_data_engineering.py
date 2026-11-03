import io
import json
import numpy as np
import torch
from PIL import Image
from app import create_app
from data_engineering import engineered_tensor, augment_batch
from model import image_tensor, load_model
from train_engineered import fit_trial, load_images
from evaluate import evaluate

CONFIG = dict(version='reflect_v1',size=128,luminance_autocontrast=True)


def test_preprocessing_and_augmentation():
    # Portrait, transparent, grayscale and one-pixel width inputs are supported.
    for image in [Image.new('RGBA',(1,80),(220,160,50,0)),
                  Image.new('L',(70,30),100), Image.new('RGB',(40,80),(220,160,50))]:
        tensor = engineered_tensor(image,CONFIG)
        assert tensor.shape == (3,128,128)
        assert tensor.dtype == torch.float32
        assert tensor.min()>=0 and tensor.max()<=1
        torch.testing.assert_close(tensor,engineered_tensor(image,CONFIG),rtol=0,atol=0)
    image = Image.new('RGB',(40,80),(220,160,50))
    reflected = engineered_tensor(image,dict(CONFIG,luminance_autocontrast=False))
    assert reflected[:,0,0].sum()>0  # reflection introduces no black letterboxing
    x = torch.linspace(0,1,3*128*128).reshape(1,3,128,128).repeat(2,1,1,1)
    original=x.clone()
    a=augment_batch(x,torch.Generator().manual_seed(42))
    b=augment_batch(x,torch.Generator().manual_seed(42))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(x,original,rtol=0,atol=0)
    assert not torch.equal(a,x)
    assert a.min()>=0 and a.max()<=1


def test_engineered_training_evaluation_serving(tmp_path):
    torch.set_num_threads(2)
    rng=np.random.default_rng(123)
    data=tmp_path/'data'
    for split in ['train','val']:
        for label in ['fake','real']:
            folder=data/split/label
            folder.mkdir(parents=True)
            for i in range(2):
                Image.fromarray(rng.integers(0,256,(40,60,3),dtype=np.uint8)).save(folder/f'{i}.png')
    excluded=['train/real/1.png']
    training=load_images(data/'train',CONFIG,excluded)
    validation=load_images(data/'val',CONFIG,excluded)
    assert len(training)==3
    assert len(validation)==4  # exclusions never remove validation or test images
    output=tmp_path/'engineered.pt'
    fit_trial(data,output,CONFIG,epochs=1,exclusions=excluded)
    model=load_model(output)
    assert model.preprocessing==CONFIG
    metrics=evaluate(data/'val',output)
    assert metrics['images']==4
    assert sum(map(sum,metrics['confusion_matrix']))==4
    path=data/'val'/'real'/'0.png'
    with Image.open(path) as image,torch.inference_mode():
        expected=model(image_tensor(image,model.preprocessing)[None]).softmax(1)[0]
    client=create_app(output).test_client()
    response=client.post('/predict',data={'image':(io.BytesIO(path.read_bytes()),'sample.png')})
    assert response.status_code==200
    assert response.json['real_score']==round(expected[1].item()*100,2)
    assert (data/'train'/'real'/'1.png').is_file()  # exclusion is non-destructive


def test_audit_flags_without_modifying_images(tmp_path):
    from audit_data import audit
    image=Image.fromarray(np.random.default_rng(12).integers(0,256,(40,40,3),dtype=np.uint8))
    for split in ['train','val']:
        folder=tmp_path/'data'/split/'real'
        folder.mkdir(parents=True)
        image.save(folder/'same.png')
    broken=tmp_path/'data'/'train'/'real'/'broken.png'
    broken.write_bytes(b'not an image')
    before=(tmp_path/'data'/'train'/'real'/'same.png').read_bytes()
    report=audit(tmp_path/'data',tmp_path/'audit.json')
    assert len(report['corrupt'])==1
    assert any(pair['cross_split'] for pair in report['near_duplicate_candidates'])
    assert (tmp_path/'data'/'train'/'real'/'same.png').read_bytes()==before
    assert broken.exists()


def test_exclusion_scope(tmp_path):
    import pytest
    from train_engineered import validate_membership
    with pytest.raises(ValueError,match='training images only'):
        validate_membership(tmp_path,['test/real/image.png'])
    with pytest.raises(ValueError,match='training images only'):
        validate_membership(tmp_path,['train/real/../../image.png'])
