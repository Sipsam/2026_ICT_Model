"""Shared folder names for the SMT ingestion and migration workflows."""

import re

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


