#!/usr/bin/env python3
"""Build a traceable SMT image/sensor manifest from English-path audit outputs.

Run from repository root:
  python scripts/build_manifest.py
  python scripts/build_manifest.py --audit outputs/data_audit_hash

No original data is changed. This DOES NOT generate detection labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

TARGET_IDS = {23: 'cold_solder', 24: 'misalignment', 25: 'short', 27: 'missing_component', 30: 'tombstone'}
NORMAL_IDS = {7, 8, 9, 11, 14}
STRUCTURAL = {
    'json_parse_error', 'missing_image', 'missing_sensor', 'invalid_filename',
    'invalid_bbox', 'invalid_category_id', 'invalid_image_metadata',
    'image_read_error', 'image_size_mismatch',
}
# These cases should not automatically become training labels.
HOLD = {
    'expected_category_missing', 'folder_status_conflict',
    'other_defect_class', 'filename_status_sensor_conflict',
    'sensor_value_mismatch', 'sensor_timestamp_mismatch',
    'sensor_length_mismatch', 'sensor_read_error',
    'sensor_sequence_missing', 'image_metadata_filename_mismatch',
    'duplicate_across_splits', 'same_filename_across_splits',
    'json_sensor_status_conflict',
}
FIELDS = [
    'sample_id', 'split', 'source_status', 'source_class', 'manufacturer',
    'image_path', 'annotation_path', 'sensor_path',
    'audit_status', 'audit_issues', 'use_for_detection', 'use_for_sensor',
    'decision', 'decision_reason', 'all_category_ids', 'defect_category_ids',
    'target_category_ids', 'target_class_names', 'target_box_count',
    'normal_box_count', 'other_defect_box_count', 'all_box_count',
    'sensor_image_id_offset', 'sensor_sequence_length',
]


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def file_by_extension(folder: Path, stem: str, extensions: tuple[str, ...]) -> Path | None:
    return next((folder / f'{stem}{ext}' for ext in extensions if (folder / f'{stem}{ext}').is_file()), None)


def relative(path: Path | None, base: Path) -> str:
    return path.relative_to(base).as_posix() if path else ''


def main():
    parser = argparse.ArgumentParser(description='Build SMT manifest (does not modify source data)')
    parser.add_argument('--root', type=Path, default=Path('data/extracted'))
    parser.add_argument('--audit', type=Path, default=Path('outputs/data_audit_hash'))
    parser.add_argument('--out', type=Path, default=Path('data/processed'))
    args = parser.parse_args()
    root, audit, out = args.root.resolve(), args.audit.resolve(), args.out.resolve()

    for path in (root, audit / 'samples.csv', audit / 'issues.csv', audit / 'summary.json'):
        if not path.exists():
            parser.error(f'Not found: {path}')
    summary = json.loads((audit / 'summary.json').read_text(encoding='utf-8-sig'))
    if not summary.get('checks', {}).get('hash_images'):
        print('WARNING: selected audit has NOT checked cross-split identical image bytes.')
    samples = read_csv(audit / 'samples.csv')
    issue_records = read_csv(audit / 'issues.csv')
    if len(samples) != summary.get('total_samples'):
        parser.error('samples.csv count differs from audit summary.json')

    issues_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in issue_records:
        issues_by_key[(item['sample'], item['split'])].add(item['issue'])

    output = []
    decisions = Counter()
    reasons = Counter()
    by_split = Counter()
    missing_files = []

    for sample in tqdm(samples, desc='Manifest 생성', unit='sample', dynamic_ncols=True):
        stem, split, folder = sample['sample'], sample['split'], sample['folder']
        key = (stem, split)
        flags = set(issues_by_key.get(key, set()))
        # English annotation folder scheme, e.g. train_annotations_soldering_defect_cold_solder.
        if '_annotations_' not in folder:
            flags.add('unexpected_annotation_folder')
        image_dir = root / folder.replace('_annotations_', '_images_', 1)
        sensor_dir = root / folder.replace('_annotations_', '_sensors_', 1)
        ann_path = root / folder / f'{stem}.json'
        img_path = file_by_extension(image_dir, stem, ('.JPG', '.jpg', '.JPEG', '.jpeg', '.PNG', '.png'))
        sensor_path = file_by_extension(sensor_dir, stem, ('.xlsx', '.csv'))
        if not ann_path.is_file() or img_path is None or sensor_path is None:
            flags.add('missing_file_on_manifest_build')
            missing_files.append(stem)

        parts = stem.split('_')
        source_status = parts[1] if len(parts) > 1 else ''
        source_class = parts[2] if len(parts) > 2 else ''
        manufacturer = parts[3] if len(parts) > 3 else ''
        cats = set()
        counts = Counter()
        sensor_offset = ''
        seq_length = ''
        if ann_path.is_file():
            try:
                data = json.loads(ann_path.read_text(encoding='utf-8-sig'))
                annotations = data.get('annotations', [])
                if not isinstance(annotations, list):
                    raise ValueError('annotations must be a list')
                for ann in annotations:
                    if not isinstance(ann, dict) or not isinstance(ann.get('category_id'), int):
                        flags.add('invalid_category_id')
                        continue
                    cid = ann['category_id']
                    cats.add(cid)
                    counts[cid] += 1
                ims = data.get('images', [])
                sgroups = data.get('sensor_data', [])
                if ims and sgroups and isinstance(ims[0], dict) and isinstance(sgroups[0], dict):
                    i = ims[0].get('id')
                    s = sgroups[0].get('image_id')
                    if isinstance(i, int) and isinstance(s, int):
                        sensor_offset = s - i
                    seq = sgroups[0].get('sensor_sequence', [])
                    if isinstance(seq, list):
                        seq_length = len(seq)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                flags.add('json_parse_error')
                missing_files.append(f'{stem}: {exc}')

        target_ids = sorted(cats & TARGET_IDS.keys())
        other_ids = sorted(cid for cid in cats if 17 <= cid <= 32 and cid not in TARGET_IDS)
        target_boxes = sum(counts[cid] for cid in TARGET_IDS)
        normal_boxes = sum(counts[cid] for cid in counts if 1 <= cid <= 16)
        other_boxes = sum(counts[cid] for cid in other_ids)

        if sample['status'] == 'invalid' or flags & STRUCTURAL or 'missing_file_on_manifest_build' in flags or 'unexpected_annotation_folder' in flags:
            decision = 'exclude'
            why = 'structural_error'
        elif flags & HOLD:
            decision = 'hold_review'
            why = ';'.join(sorted(flags & HOLD))
        elif flags - {'mixed_normal_defect_boxes'}:
            decision = 'hold_review'
            why = 'unrecognized_issue:' + ';'.join(sorted(flags - {'mixed_normal_defect_boxes'}))
        else:
            # Normal boxes are not defect targets. A mixed image can still be valid for object detection.
            decision = 'candidate'
            why = 'mixed_objects_allowed' if 'mixed_normal_defect_boxes' in flags else 'audit_passed'

        decisions[decision] += 1
        reasons[why] += 1
        by_split[(split, decision)] += 1
        output.append({
            'sample_id': stem, 'split': split, 'source_status': source_status,
            'source_class': source_class, 'manufacturer': manufacturer,
            'image_path': relative(img_path, root.parent.parent),
            'annotation_path': relative(ann_path, root.parent.parent),
            'sensor_path': relative(sensor_path, root.parent.parent),
            'audit_status': sample['status'], 'audit_issues': ';'.join(sorted(flags)),
            'use_for_detection': 'candidate' if decision == 'candidate' else 'no',
            # Sensor classification target policy is NOT finalized; no automatic sensor training approval.
            'use_for_sensor': 'pending_label_policy',
            'decision': decision, 'decision_reason': why,
            'all_category_ids': ';'.join(map(str, sorted(cats))),
            'defect_category_ids': ';'.join(map(str, sorted(cid for cid in cats if 17 <= cid <= 32))),
            'target_category_ids': ';'.join(map(str, target_ids)),
            'target_class_names': ';'.join(TARGET_IDS[cid] for cid in target_ids),
            'target_box_count': target_boxes, 'normal_box_count': normal_boxes,
            'other_defect_box_count': other_boxes, 'all_box_count': sum(counts.values()),
            'sensor_image_id_offset': sensor_offset, 'sensor_sequence_length': seq_length,
        })

    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.csv'
    with manifest_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    report = {
        'total_samples': len(output), 'by_decision': dict(decisions),
        'by_split_decision': {f'{s}/{d}': count for (s, d), count in sorted(by_split.items())},
        'decision_reasons': dict(reasons), 'audit_source': str(audit),
        'source_root': str(root),
        'warnings': [
            'candidate means eligible for later conversion review, not fully verified ground truth.',
            'Mixed normal/defect objects are allowed only when no other warning is present.',
            'Detection labels are NOT generated here. Other unselected defect classes remain on hold.',
            'Sensor classification labels are NOT finalized; defect_detected must never be used as an input feature.',
            'Paths are relative to repository root. Keep extracted images/annotations/sensors in place.',
        ],
    }
    report_path = out / 'manifest_summary.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n=== Manifest 완료 ===')
    print('총 샘플:', len(output))
    print('판정:', dict(decisions))
    print('분할:', report['by_split_decision'])
    print('출력:', manifest_path)
    print('요약:', report_path)
    if missing_files:
        print('WARNING: 누락/읽기 실패:', len(missing_files), '건 (exclude 처리)')


if __name__ == '__main__':
    main()
