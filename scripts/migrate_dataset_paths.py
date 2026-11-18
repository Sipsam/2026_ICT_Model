#!/usr/bin/env python3
"""Rename extracted AI Hub SMT folder names to stable ASCII names. No file content changes.

Usage:
  python scripts/migrate_dataset_paths.py              # dry run
  python scripts/migrate_dataset_paths.py --apply      # actual directory renames
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Keep direct script execution available from a checkout without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ict_pipeline.data.naming import DEFECTS as DEFECTS, PATTERN as PATTERN, PREFIXES, translated


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
