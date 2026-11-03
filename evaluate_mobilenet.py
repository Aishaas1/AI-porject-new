import argparse
import json
from pathlib import Path
from train_mobilenet import load_split, metrics_for
from mobilenet import keras

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/goldnet/test')
    parser.add_argument('--model', default='artifacts/gold_mobilenet.keras')
    parser.add_argument('--output', default='reports/mobilenet_test_metrics.json')
    args = parser.parse_args()
    x, y, _ = load_split(args.data)
    model = keras.models.load_model(args.model, compile=False)
    metrics = metrics_for(model, x, y)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
