"""Select a trained artifact by validation only; does not read test results."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch


def select_best(trials_path, output):
    trials=json.loads(Path(trials_path).read_text())
    if not trials:
        raise ValueError('No completed trials')
    best=max(trials,key=lambda trial: trial['validation_accuracy'])
    source=Path(best['path'])
    checkpoint=torch.load(source,map_location='cpu',weights_only=True)
    if checkpoint['metrics']['validation_accuracy']!=best['validation_accuracy']:
        raise ValueError('Trial report does not match saved checkpoint')
    if checkpoint['metrics']['train_images']!=1401 or checkpoint['metrics']['val_images']!=302:
        raise ValueError('Unexpected GoldNet split membership')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if source.resolve()!=output.resolve():
        shutil.copy2(source,output)
    report=dict(criterion='maximum validation accuracy; ties prefer earlier/simpler trial',
                selected_trial=source.stem,checkpoint=str(output),
                checkpoint_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                validation_accuracy=best['validation_accuracy'],
                preprocessing=best['preprocessing'],augmentation=best['augmentation'],
                metrics=checkpoint['metrics'],trials=trials)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    Path('reports/engineering_selection.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--trials',default='reports/engineering_trials.json')
    parser.add_argument('--output',default='artifacts/gold_engineered.pt')
    args=parser.parse_args()
    select_best(args.trials,args.output)
