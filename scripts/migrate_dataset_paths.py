#!/usr/bin/env python3
"""Rename extracted AI Hub SMT folder names to stable ASCII names. No file content changes.

Usage:
  python scripts/migrate_dataset_paths.py              # dry run
  python scripts/migrate_dataset_paths.py --apply      # actual directory renames
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFECTS = {
    '냉납': 'cold_solder',
    '밀림': 'misalignment',
    '쇼트': 'short',
    '미삽': 'missing_component',
    '일어섬': 'tombstone',
}
PREFIXES = {
    'TL_': ('train', 'annotations'),
    'VL_': ('validation', 'annotations'),
    'TS_이미지데이터_': ('train', 'images'),
    'VS_이미지데이터_': ('validation', 'images'),
    'TS_센서데이터_': ('train', 'sensors'),
    'VS_센서데이터_': ('validation', 'sensors'),
}
PATTERN = re.compile(r'^납땜공정_(정상|불량)_(\d+)\.([^/\\]+)$')


def translated(name: str) -> str | None:
    for prefix, (split, kind) in PREFIXES.items():
        if not name.startswith(prefix):
            continue
        match = PATTERN.fullmatch(name[len(prefix):])
        if match is None:
            return None
        state_ko, _number, label_ko = match.groups()
        if label_ko not in DEFECTS:
            return None
        state = 'normal' if state_ko == '정상' else 'defect'
        return f'{split}_{kind}_soldering_{state}_{DEFECTS[label_ko]}'
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description='Migrate extracted Korean SMT folder names to English.')
    parser.add_argument('--root', type=Path, default=Path('data/extracted'))
    parser.add_argument('--apply', action='store_true', help='Perform the folder renames; default is preview only.')
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f'Missing folder: {root}')

    sources = sorted(p for p in root.iterdir() if p.is_dir())
    mappings = [(p, root / new) for p in sources if (new := translated(p.name))]
    unknown = [p for p in sources if any(p.name.startswith(k) for k in PREFIXES) and translated(p.name) is None]
    destinations = [dst for _, dst in mappings]
    collision = [str(dst) for dst in destinations if dst.exists()]  # never overwrite
    if len(destinations) != len(set(destinations)):
        parser.error('Two source folders map to the same destination. No changes made.')
    if collision:
        parser.error(f'Destination already exists; no changes made: {collision[:3]}')
    if unknown:
        parser.error('Unrecognized dataset directories; no changes made: ' + ', '.join(p.name for p in unknown[:10]))
    if not mappings:
        print('No Korean folders remain to migrate. Already migrated, or folder layout differs.')
        return

    print(f'Folders to rename: {len(mappings)}')
    for src, dst in mappings:
        print(f'{src.name}  ->  {dst.name}')
    if not args.apply:
        print('\nPREVIEW ONLY. Run again with --apply to rename.')
        return

    done = []
    try:
        for src, dst in mappings:
            src.rename(dst)
            done.append((src, dst))
    except Exception:
        print('Rename failed; reverting renamed folders...')
        for src, dst in reversed(done):
            if dst.exists() and not src.exists():
                dst.rename(src)
        raise
    print(f'\nDone: {len(done)} directories renamed. File contents unchanged.')
    print('Next: replace the validator/visualization scripts with English-path versions and rerun audit.')


if __name__ == '__main__':
    main()
