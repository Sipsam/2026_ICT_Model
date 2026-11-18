"""Build a traceable detection manifest from the SMT audit."""

from collections.abc import Mapping
import json
from collections import Counter, defaultdict
from pathlib import Path

from ...pipeline.store import fingerprint_path
from ..model import Progress
from .common import Row, read_csv, require_separate_output, safe_component, split_fingerprint, write_csv, write_json
from .policy import HOLD, STRUCTURAL, TARGET_IDS, label_policy

FIELDS = [
    'sample_id', 'split', 'source_status', 'source_class', 'manufacturer',
    'image_path', 'annotation_path', 'sensor_path',
    'audit_status', 'audit_issues', 'use_for_detection', 'use_for_sensor',
    'decision', 'decision_reason', 'all_category_ids', 'defect_category_ids',
    'target_category_ids', 'target_class_names', 'target_box_count',
    'normal_box_count', 'other_defect_box_count', 'all_box_count',
    'sensor_image_id_offset', 'sensor_sequence_length',
]


def file_by_extension(folder: Path, stem: str, extensions: tuple[str, ...]) -> Path | None:
    return next((folder / f'{stem}{ext}' for ext in extensions if (folder / f'{stem}{ext}').is_file()), None)


def relative(path: Path | None, base: Path) -> str:
    return path.relative_to(base).as_posix() if path else ''



def build_manifest(
    root: Path, audit: Path, out: Path, *, path_base: Path | None = None,
    progress: Progress | None = None, source_fingerprint: str | None = None,
    audit_fingerprints: Mapping[str, str] | None = None,
) -> dict[str, object]:
    root, audit, out = root.resolve(), audit.resolve(), out.resolve()
    base = path_base.resolve() if path_base is not None else root
    require_separate_output(root, out)
    require_separate_output(audit, out)
    for path in (root, audit / 'samples.csv', audit / 'issues.csv', audit / 'summary.json'):
        if not path.exists():
            raise ValueError(f'Not found: {path}')
    summary = json.loads((audit / 'summary.json').read_text(encoding='utf-8-sig'))
    samples = read_csv(audit / 'samples.csv')
    issue_records = read_csv(audit / 'issues.csv')
    if len(samples) != summary.get('total_samples'):
        raise ValueError('samples.csv count differs from audit summary.json')

    source_fingerprint = source_fingerprint or fingerprint_path(root)
    if summary.get('source_fingerprint') not in (None, source_fingerprint):
        raise ValueError('The dataset changed since the audit. Run the audit again.')
    if audit_fingerprints is None:
        audit_fingerprints = {name: fingerprint_path(audit / name) for name in ('samples.csv', 'issues.csv', 'summary.json')}

    issues_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in issue_records:
        issues_by_key[(item['sample'], item['split'])].add(item['issue'])

    output: list[Row] = []
    decisions = Counter()
    reasons = Counter()
    by_split = Counter()

    if progress:
        progress(0, len(samples))
    for done, sample in enumerate(samples, 1):
        stem, split, folder = sample['sample'], sample['split'], sample['folder']
        if not safe_component(stem) or not safe_component(folder):
            raise ValueError('Audit contains an invalid sample or folder name.')
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
                if not isinstance(data, dict):
                    raise ValueError('expected a JSON object')
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
                if isinstance(ims, list) and ims and isinstance(sgroups, list) and sgroups and isinstance(ims[0], dict) and isinstance(sgroups[0], dict):
                    i = ims[0].get('id')
                    s = sgroups[0].get('image_id')
                    if isinstance(i, int) and isinstance(s, int):
                        sensor_offset = s - i
                    seq = sgroups[0].get('sensor_sequence', [])
                    if isinstance(seq, list):
                        seq_length = len(seq)
            except (OSError, ValueError):
                flags.add('json_parse_error')

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
            'image_path': relative(img_path, base),
            'annotation_path': relative(ann_path, base),
            'sensor_path': relative(sensor_path, base),
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
        if progress:
            progress(done, len(samples))

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'manifest.csv', output, FIELDS)
    report: dict[str, object] = {
        'source_fingerprint': source_fingerprint,
        'split_fingerprint': split_fingerprint(samples),
        'audit_fingerprints': dict(audit_fingerprints),
        'audit_checks': summary.get('checks', {}),
        'label_policy': label_policy(),
        'path_base': 'dataset_root' if base == root else 'repository_root',
        'total_samples': len(output), 'by_decision': dict(decisions),
        'by_split_decision': {f'{s}/{d}': count for (s, d), count in sorted(by_split.items())},
        'decision_reasons': dict(reasons), 'audit_source': str(audit),
        'path_root': str(base),
        'source_root': str(root),
        'warnings': [
            'candidate means eligible for later conversion review, not fully verified ground truth.',
            'Mixed normal/defect objects are allowed only when no other warning is present.',
            'Detection labels are NOT generated here. Other unselected defect classes remain on hold.',
            'Sensor classification labels are NOT finalized; defect_detected must never be used as an input feature.',
            'Resolve relative paths using the recorded path base.',
        ],
    }
    write_json(out / 'manifest_summary.json', report)
    return report
