import argparse
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model import CLASSES, GoldCNN, image_tensor, load_model

class GoldDataset(Dataset):
    def __init__(self, root, preprocessing=None):
        self.samples = []
        self.cache = {}
        self.preprocessing = preprocessing
        for label, name in enumerate(CLASSES):
            paths = sorted(p for p in (Path(root)/name).glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
            if not paths:
                raise ValueError(f'لا توجد صور في {Path(root)/name}')
            self.samples.extend((p, label) for p in paths)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        path, label = self.samples[index]
        if index not in self.cache:
            with Image.open(path) as image:
                self.cache[index] = image_tensor(image, self.preprocessing)
        return self.cache[index], label

def train(data, output, epochs=10, batch_size=16, learning_rate=.001, resume=None):
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch-size must be positive')
    torch.manual_seed(42)
    training, validation = GoldDataset(Path(data)/'train'), GoldDataset(Path(data)/'val')
    hashes = {hashlib.sha256(p.read_bytes()).digest() for p, _ in training.samples}
    if any(hashlib.sha256(p.read_bytes()).digest() in hashes for p, _ in validation.samples):
        raise ValueError('توجد صور مكررة بين التدريب والتحقق')
    if learning_rate <= 0:
        raise ValueError('learning-rate must be positive')
    model = load_model(resume) if resume else GoldCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = -1
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.
        for images, labels in DataLoader(training, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(labels)
        model.eval()
        matrix = torch.zeros((2, 2), dtype=torch.int64)
        with torch.inference_mode():
            for images, labels in DataLoader(validation, batch_size=batch_size):
                for actual, predicted in zip(labels, model(images).argmax(1)):
                    matrix[actual, predicted] += 1
        accuracy = matrix.diag().sum().item()/len(validation)
        metrics = dict(epoch=epoch+1, loss=loss_sum/len(training), validation_accuracy=accuracy,
                       confusion_matrix=matrix.tolist(), classes=CLASSES,
                       train_images=len(training), val_images=len(validation))
        print(json.dumps(metrics), flush=True)
        if accuracy > best:
            best = accuracy
            torch.save(dict(state_dict=model.state_dict(), classes=CLASSES, metrics=metrics), output)
            output.with_suffix('.json').write_text(json.dumps(metrics, indent=2))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='تدريب CNN لتصنيف صور الذهب')
    parser.add_argument('--data', default='data')
    parser.add_argument('--output', default='artifacts/gold.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--resume')
    args = parser.parse_args()
    try:
        train(args.data, args.output, args.epochs, args.batch_size, args.learning_rate, args.resume)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
