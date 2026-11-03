import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from mobilenet import keras, build_model, enable_fine_tuning, image_array, CLASSES

def load_split(directory):
    images, labels, hashes = [], [], set()
    for label, name in enumerate(CLASSES):
        paths = sorted(p for p in (Path(directory)/name).glob('*')
                       if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
        if not paths:
            raise ValueError(f'No images in {Path(directory)/name}')
        for path in paths:
            hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
            with Image.open(path) as image:
                images.append(image_array(image))
            labels.append(label)
    return np.stack(images), np.array(labels, dtype=np.int64), hashes

def metrics_for(model, images, labels, batch_size=32):
    predictions = model.predict(images, batch_size=batch_size, verbose=0).argmax(1)
    matrix = np.zeros((2,2), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    tp = int(matrix[1,1])
    precision = tp/max(1, int(matrix[:,1].sum()))
    recall = tp/max(1, int(matrix[1,:].sum()))
    return dict(images=len(labels), classes=CLASSES, confusion_matrix=matrix.tolist(),
                accuracy=float(np.trace(matrix)/len(labels)), real_precision=precision,
                real_recall=recall, real_f1=2*precision*recall/max(1e-12,precision+recall))

class BestCheckpoint(keras.callbacks.Callback):
    """One validation criterion shared by frozen and fine-tuning stages."""
    def __init__(self, output):
        super().__init__()
        self.output = Path(output)
        self.best = -1.
        self.history = []
        self.stage = 'head'
    def on_epoch_end(self, epoch, logs=None):
        record = dict(stage=self.stage, epoch=epoch+1,
                      **{k:float(v) for k,v in (logs or {}).items()})
        self.history.append(record)
        print(json.dumps(record), flush=True)
        if record['val_accuracy'] > self.best:
            self.best = record['val_accuracy']
            self.model.save(self.output)
            self.output.with_suffix('.json').write_text(json.dumps(record, indent=2))
        self.output.with_name(self.output.stem+'_history.json').write_text(json.dumps(self.history, indent=2))

def train(data, output, head_epochs=3, fine_epochs=7, batch_size=32):
    if min(head_epochs, fine_epochs, batch_size) < 1:
        raise ValueError('Epochs and batch size must be positive')
    keras.utils.set_random_seed(42)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print('Loading training and validation images (test split is not loaded).', flush=True)
    x, y, train_hashes = load_split(Path(data)/'train')
    vx, vy, val_hashes = load_split(Path(data)/'val')
    if train_hashes & val_hashes:
        raise ValueError('Duplicate images across training and validation')
    model, base = build_model()
    callback = BestCheckpoint(output)
    print('Stage 1: backbone frozen, head learning rate 0.001', flush=True)
    model.fit(x, y, validation_data=(vx,vy), epochs=head_epochs,
              batch_size=batch_size, callbacks=[callback], verbose=2)
    # Start fine tuning from the best frozen-stage state.
    model.set_weights(keras.models.load_model(output, compile=False).get_weights())
    flags = enable_fine_tuning(model, base)
    output.with_name(output.stem+'_layers.json').write_text(json.dumps(flags, indent=2))
    callback.stage = 'fine_tuning'
    print('Stage 2: last 30 layers only, BN frozen, recompiled with learning rate 1e-5', flush=True)
    model.fit(x, y, validation_data=(vx,vy), epochs=fine_epochs,
              batch_size=batch_size, callbacks=[callback], verbose=2)
    selected = keras.models.load_model(output, compile=False)
    metrics = metrics_for(selected, vx, vy, batch_size)
    output.with_name(output.stem+'_validation.json').write_text(json.dumps(metrics, indent=2))
    return metrics

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/goldnet')
    p.add_argument('--output', default='artifacts/gold_mobilenet.keras')
    p.add_argument('--head-epochs', type=int, default=3)
    p.add_argument('--fine-epochs', type=int, default=7)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    train(args.data, args.output, args.head_epochs, args.fine_epochs, args.batch_size)
