"""Non-destructive image quality and near-duplicate audit; never relabels images."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageOps


def audit(root, output):
    root = Path(root)
    records, corrupt = [], []
    for split in ('train', 'val', 'test'):
        for label in ('fake', 'real'):
            for path in sorted((root/split/label).glob('*')):
                if path.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:
                    continue
                try:
                    with Image.open(path) as image:
                        image = ImageOps.exif_transpose(image).convert('RGB')
                        gray = np.asarray(image.convert('L').resize((128,128)), dtype=np.float32)
                        small = np.asarray(image.convert('L').resize((17,16)), dtype=np.int16)
                        bits = np.packbits((small[:,1:] > small[:,:-1]).ravel())
                        # dHash is a screening signal, not proof that two items are identical.
                        record = dict(path=str(path.relative_to(root)), split=split, label=label,
                                      width=image.width, height=image.height, brightness=float(gray.mean()),
                                      contrast=float(gray.std()),
                                      edge_energy=float(np.abs(np.diff(gray,axis=0)).mean()+np.abs(np.diff(gray,axis=1)).mean()),
                                      dhash=bits.tobytes().hex())
                        records.append(record)
                except (OSError, ValueError) as exc:
                    corrupt.append(dict(path=str(path.relative_to(root)),error=type(exc).__name__))
    hashes = [int(r['dhash'],16) for r in records]
    near = []
    for i, a in enumerate(records):
        for j in range(i):
            distance = (hashes[i]^hashes[j]).bit_count()
            if distance <= 8:
                b = records[j]
                near.append(dict(first=b['path'],second=a['path'],distance=distance,
                                 cross_split=a['split']!=b['split'], conflicting_labels=a['label']!=b['label']))
    counts = Counter(f"{r['split']}/{r['label']}" for r in records)
    flags = [r['path'] for r in records if min(r['width'],r['height'])<128 or
             r['brightness']<25 or r['brightness']>235 or r['contrast']<10]
    report=dict(counts=dict(counts),corrupt=corrupt,quality_flags_for_review=flags,
                near_duplicate_candidates=near,images=records,
                note='Quality thresholds and 256-bit dHash distance <=8 flag review candidates only; no images are removed or relabelled.')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(images=len(records),counts=dict(counts),corrupt=len(corrupt),
                          quality_flags=len(flags),near_duplicate_candidates=len(near),
                          cross_split_candidates=sum(r['cross_split'] for r in near)),indent=2))
    return report

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/goldnet')
    parser.add_argument('--output',default='reports/data_audit.json')
    args=parser.parse_args()
    audit(args.data,args.output)
