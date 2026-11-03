import io
import json
import numpy as np
import pytest
import torch
from PIL import Image
from app import create_app
from train import train

def png(seed):
    buffer = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(buffer, format='PNG')
    return buffer.getvalue()

def test_untrained_app(tmp_path):
    client = create_app(tmp_path/'absent.pt').test_client()
    assert client.get('/').status_code == 200
    assert client.get('/health').json['model_ready'] is False
    assert client.post('/predict').status_code == 503

def test_training_and_upload(tmp_path):
    torch.set_num_threads(2)
    for i, (split, label) in enumerate((('train','fake'),('train','real'),('val','fake'),('val','real'))):
        directory = tmp_path/'data'/split/label
        directory.mkdir(parents=True)
        (directory/'photo.png').write_bytes(png(i))
    path = tmp_path/'gold.pt'
    train(tmp_path/'data', path, epochs=1, batch_size=2)
    metrics = json.loads(path.with_suffix('.json').read_text())
    assert sum(map(sum, metrics['confusion_matrix'])) == 2
    client = create_app(path).test_client()
    assert client.get('/health').json['model_ready'] is True
    response = client.post('/predict', data={'image':(io.BytesIO(png(20)), 'test.png')})
    assert response.status_code == 200
    assert response.json['label'] in ['حقيقي','مزيف']
    assert abs(response.json['real_score']+response.json['fake_score']-100) < .02
    assert client.post('/predict').status_code == 400
    assert client.post('/predict', data={'image':(io.BytesIO(b'invalid'), 'bad.png')}).status_code == 400
    assert client.post('/predict', data={'image':(io.BytesIO(b'x'*9*1024*1024), 'large.png')}).status_code == 413
    resumed = tmp_path/'resumed.pt'
    original_bytes = path.read_bytes()
    train(tmp_path/'data', resumed, epochs=1, batch_size=2,
          learning_rate=1e-4, resume=path)
    assert resumed.is_file()
    assert path.read_bytes() == original_bytes
    (tmp_path/'data'/'val'/'fake'/'photo.png').write_bytes((tmp_path/'data'/'train'/'fake'/'photo.png').read_bytes())
    with pytest.raises(ValueError, match='مكررة'):
        train(tmp_path/'data', path, epochs=1)

def test_goldnet_split(tmp_path):
    from prepare_goldnet import prepare
    source = tmp_path/'source'
    source.mkdir()
    (source/'LICENSE-DATA').write_text('fixture')
    for label, offset in [('fake', 0), ('real', 100)]:
        directory = source/'gold'/label
        directory.mkdir(parents=True)
        for i in range(10):
            (directory/f'{i}.png').write_bytes(png(offset+i))
        (directory/'duplicate.png').write_bytes((directory/'0.png').read_bytes())
    destination = tmp_path/'prepared'
    prepare(source, destination)
    manifest = json.loads((destination/'manifest.json').read_text())
    assert manifest['duplicate_images_removed'] == 2
    hashes = [r['pixel_sha256'] for r in manifest['images']]
    assert len(hashes) == len(set(hashes)) == 20
    assert sum(manifest['counts'].values()) == 20
    assert all(r['split'] in {'train', 'val', 'test'} for r in manifest['images'])
    with pytest.raises(ValueError, match='exists'):
        prepare(source, destination)
