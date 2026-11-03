"""Prepare a reproducible stratified split; never overwrite an existing dataset."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from PIL import Image, ImageOps

SOURCE_URL = 'https://github.com/zobeirraisi/GoldNet'
SOURCE_COMMIT = 'f206c941a39d3126b6ce867e01abfad87e10f525'

def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('Destination already exists; choose a new directory')
    records, seen = [], {}
    duplicates = 0
    for label in ['fake', 'real']:
        for path in sorted((source/'gold'/label).glob('*')):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
                continue
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                digest = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError('Conflicting labels for identical images')
                duplicates += 1
                continue
            seen[digest] = label
            records.append(dict(source=str(path.relative_to(source)), label=label,
                                pixel_sha256=digest, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rng = random.Random(42)
    counts = {}
    for label in ['fake', 'real']:
        items = [r for r in records if r['label'] == label]
        if len(items) < 10:
            raise ValueError('Not enough images')
        rng.shuffle(items)
        a, b = int(.70*len(items)), int(.85*len(items))
        for split, subset in [('train', items[:a]), ('val', items[a:b]), ('test', items[b:])]:
            counts[f'{split}/{label}'] = len(subset)
            for record in subset:
                record['split'] = split
    destination.mkdir(parents=True)
    for record in records:
        path = source/record['source']
        target = destination/record['split']/record['label']/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(source/'LICENSE-DATA', destination/'LICENSE-DATA')
    manifest = dict(source=SOURCE_URL, commit=SOURCE_COMMIT, seed=42, counts=counts,
                    duplicate_images_removed=duplicates, images=records)
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='images'}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', default='data/goldnet')
    args = parser.parse_args()
    prepare(args.source, args.destination)
