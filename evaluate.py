"""Evaluate a selected checkpoint once on an independent held-out split."""
import argparse
import json
import torch
from torch.utils.data import DataLoader
from model import CLASSES, load_model
from train import GoldDataset

def evaluate(data, model_path):
    model = load_model(model_path)
    dataset = GoldDataset(data, getattr(model, 'preprocessing', None))
    matrix = torch.zeros((2, 2), dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=32):
            predictions = (model(images).softmax(1)[:,1] > getattr(model,'decision_threshold',.5)).long()
            for actual, predicted in zip(labels, predictions):
                matrix[actual, predicted] += 1
    tp = matrix[1,1].item()
    precision = tp/max(1, matrix[:,1].sum().item())
    recall = tp/max(1, matrix[1,:].sum().item())
    return dict(images=len(dataset), classes=CLASSES, confusion_matrix=matrix.tolist(),
                accuracy=matrix.diag().sum().item()/len(dataset),
                real_precision=precision, real_recall=recall,
                real_f1=2*precision*recall/max(1e-12, precision+recall))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', default='artifacts/gold.pt')
    parser.add_argument('--output', default='artifacts/test_metrics.json')
    args = parser.parse_args()
    metrics = evaluate(args.data, args.model)
    from pathlib import Path
    Path(args.output).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
