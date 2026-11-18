"""Draw bounding boxes for issues from the English-path validator.

Usage:
  python scripts/visualize_issues.py --issues outputs/data_audit_english/issues.csv
  python scripts/visualize_issues.py --issues outputs/data_audit_english/issues.csv --kind expected_category_missing
"""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm

CODE_NAMES = {'CC': 'cold_solder', 'ML': 'misalignment', 'SH': 'short',
              'MS': 'missing_component', 'ST': 'tombstone'}
LABEL_NAMES = {
    7: 'normal_cold_solder', 8: 'normal_misalignment', 9: 'normal_short',
    11: 'normal_missing_component', 14: 'normal_tombstone',
    17: 'defect_missing_solder', 20: 'defect_solder_ball', 23: 'defect_cold_solder',
    24: 'defect_misalignment', 25: 'defect_short', 26: 'defect_wrong_component',
    27: 'defect_missing_component', 29: 'defect_flipped', 30: 'defect_tombstone'
}
COLORS = ('#ff4141', '#00c8ff', '#ffc400', '#b36bff', '#00cc88')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data/extracted'))
    parser.add_argument('--issues', type=Path, default=Path('outputs/data_audit_english/issues.csv'))
    parser.add_argument('--out', type=Path, default=Path('outputs/data_audit_english/review'))
    parser.add_argument('--kind', default='expected_category_missing',
                        help='Issue category to visualize; use all for every kind')
    args = parser.parse_args()
    with args.issues.open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    if args.kind != 'all':
        records = [r for r in records if r['issue'] == args.kind]
    # De-duplicate samples with multiple flags; split is included to avoid collisions.
    unique = {(r['sample'], r['split']): r for r in records}
    args.out.mkdir(parents=True, exist_ok=True)
    results = []

    for (stem, split), record in tqdm(sorted(unique.items()), desc='Visualizing', unit='image'):
        try:
            fields = stem.split('_')
            state = {'NOR': 'normal', 'DEF': 'defect'}[fields[1]]
            name = CODE_NAMES[fields[2]]
            suffix = f'{split}_{{kind}}_soldering_{state}_{name}'
            label_path = args.root / suffix.format(kind='annotations') / f'{stem}.json'
            image_dir = args.root / suffix.format(kind='images')
            image_path = next((p for ext in ('.JPG','.jpg','.JPEG','.jpeg')
                               if (p := image_dir / f'{stem}{ext}').is_file()), None)
            if image_path is None:
                raise FileNotFoundError(f'Image not found in {image_dir}')
            data = json.loads(label_path.read_text(encoding='utf-8-sig'))
            annotations = data.get('annotations', [])
            with Image.open(image_path) as im:
                img = im.convert('RGB')
            w, h = img.size
            lines = [f'{split}: {stem}',
                     f'Expected: {state}_{name}',
                     f'Issue: {record["issue"]} ({record["detail"]})']
            for i, ann in enumerate(annotations):
                cid = ann.get('category_id')
                lines.append(f'Box {i+1}: [{cid}] {LABEL_NAMES.get(cid, "unknown")}')
            outimg = Image.new('RGB', (w, h + len(lines)*20 + 12), '#191919')
            outimg.paste(img, (0, 0))
            draw = ImageDraw.Draw(outimg)
            for i, ann in enumerate(annotations):
                box = ann.get('bbox', [])
                if not isinstance(box, list) or len(box) != 4:
                    continue
                x, y, bw, bh = box
                color = COLORS[i % len(COLORS)]
                draw.rectangle((max(0,x), max(0,y), min(w,x+bw), min(h,y+bh)),
                               outline=color, width=3)
                draw.text((max(0,x)+3, max(0,y)+3), f'#{i+1} id={ann.get("category_id")}',
                          fill=color, stroke_width=1, stroke_fill='black')
            for i, line in enumerate(lines):
                draw.text((6, h+6+i*20), line[:100], fill='white')
            target = args.out / f'{split}_{stem}.png'
            outimg.save(target)
            results.append((stem, split, str(target), 'ok'))
        except Exception as exc:
            results.append((stem, split, '', f'error: {exc}'))
    with (args.out / 'review_index.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'split', 'review_image', 'result'])
        writer.writerows(results)
    print(f'Generated {sum(r[-1] == "ok" for r in results)}/{len(results)} images: {args.out}')


if __name__ == '__main__':
    main()
