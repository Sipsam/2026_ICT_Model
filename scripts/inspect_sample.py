"""Inspect one English-path SMT sample. From repository root: python scripts/inspect_sample.py"""
from pathlib import Path
import json
from pprint import pprint
import pandas as pd

root = Path('data/extracted')
stem = 'SD_DEF_CC_A_20250807-090340_00001'
image = root / 'train_images_soldering_defect_cold_solder' / (stem + '.JPG')
label = root / 'train_annotations_soldering_defect_cold_solder' / (stem + '.json')
sensor = root / 'train_sensors_soldering_defect_cold_solder' / (stem + '.xlsx')
print('File existence:', {'image': image.is_file(), 'json': label.is_file(), 'sensor': sensor.is_file()})
with label.open(encoding='utf-8-sig') as f:
    data = json.load(f)
for key, values in data.items():
    print(f'\n{key}: {len(values)}')
    if values:
        pprint(values[0])
print('\nSensor XLSX:')
print(pd.read_excel(sensor).head().to_string(index=False))
