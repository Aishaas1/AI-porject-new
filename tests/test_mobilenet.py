"""Offline structural tests use random weights, never claim ImageNet accuracy."""
import io
import numpy as np
import torch
from PIL import Image
from app import create_app
from mobilenet import keras, build_model, enable_fine_tuning, image_array


def test_freezing_fine_tuning_and_serving(tmp_path):
    torch.set_num_threads(2)
    keras.utils.set_random_seed(42)
    model, base = build_model(weights=None)
    assert base.trainable is False
    assert not base.trainable_weights
    assert model.output_shape == (None, 2)
    assert model.input_shape == (None, 224, 224, 3)
    assert base.output_shape[-1] == 1280  # No ImageNet classifier head.
    old_optimizer = model.optimizer
    flags = enable_fine_tuning(model, base)
    assert base.trainable is True
    assert all(not layer.trainable for layer in base.layers[:-30])
    assert all(layer.trainable == (not isinstance(layer, keras.layers.BatchNormalization))
               for layer in base.layers[-30:])
    assert all(not layer.trainable for layer in base.layers
               if isinstance(layer, keras.layers.BatchNormalization))
    assert old_optimizer is not model.optimizer
    assert np.isclose(float(keras.ops.convert_to_numpy(model.optimizer.learning_rate)), 1e-5)
    assert len(flags) == len(base.layers)
    frozen = [(weight, weight.numpy().copy()) for layer in base.layers
              if not layer.trainable for weight in layer.weights]
    trainable = [(weight, weight.numpy().copy()) for weight in base.trainable_weights]
    x = np.random.default_rng(42).integers(0,256,(2,224,224,3),dtype=np.uint8)
    loss = model.train_on_batch(x, np.array([0,1]))
    assert np.isfinite(loss).all()
    assert all(np.array_equal(weight.numpy(), before) for weight, before in frozen)
    assert any(not np.array_equal(weight.numpy(), before) for weight, before in trainable)
    expected = keras.ops.convert_to_numpy(model(x[:1], training=False))
    checkpoint = tmp_path/'offline_test.keras'
    model.save(checkpoint)
    restored = keras.models.load_model(checkpoint, compile=False)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(restored(x[:1], training=False)), expected, atol=1e-6)
    client = create_app(checkpoint).test_client()
    assert client.get('/health').json['architecture'] == 'MobileNetV2'
    buffer = io.BytesIO()
    Image.fromarray(x[0]).save(buffer, format='PNG')
    response = client.post('/predict', data={'image':(io.BytesIO(buffer.getvalue()), 'sample.png')})
    assert response.status_code == 200
    assert abs(response.json['real_score']+response.json['fake_score']-100) < .02
    assert client.post('/predict', data={'image':(io.BytesIO(b'broken'), 'sample.png')}).status_code == 400


def test_preprocessing_is_inside_model():
    image = Image.new('RGB', (30,70), (255,0,0))
    array = image_array(image)
    assert array.shape == (224,224,3)
    assert array.dtype == np.uint8
    # Scaling must happen exactly once, in the persisted Keras graph.
    scaling = keras.layers.Rescaling(1./127.5, offset=-1.)
    values = keras.ops.convert_to_numpy(scaling(np.array([0,127.5,255], dtype='float32')))
    np.testing.assert_allclose(values, [-1,0,1], atol=1e-6)


def test_two_stage_training_pipeline(tmp_path, monkeypatch):
    import train_mobilenet
    torch.set_num_threads(2)
    monkeypatch.setattr(train_mobilenet, 'build_model', lambda: build_model(weights=None))
    rng = np.random.default_rng(7)
    for split in ('train','val'):
        for label in ('fake','real'):
            folder = tmp_path/'data'/split/label
            folder.mkdir(parents=True)
            Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(folder/'sample.png')
    output = tmp_path/'model.keras'
    metrics = train_mobilenet.train(tmp_path/'data', output, head_epochs=1, fine_epochs=1, batch_size=2)
    import json
    history = json.loads(output.with_name('model_history.json').read_text())
    assert [record['stage'] for record in history] == ['head','fine_tuning']
    assert output.is_file()
    assert metrics['images'] == 2
    assert sum(map(sum, metrics['confusion_matrix'])) == 2
