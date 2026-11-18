"""Versioned label and split policy for AI Hub SMT dataset 71900."""

POLICY_VERSION = "aihub-smt-71900-v1"
CLASSES = {
    "CC": (7, 23, "cold_solder"),
    "ML": (8, 24, "misalignment"),
    "SH": (9, 25, "short"),
    "MS": (11, 27, "missing_component"),
    "ST": (14, 30, "tombstone"),
}
TARGET_IDS = {defect: name for _, defect, name in CLASSES.values()}
NORMAL_IDS = {normal for normal, _, _ in CLASSES.values()}
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


NAMES = list(TARGET_IDS.values())
CLASS_MAP = {source_id: index for index, source_id in enumerate(TARGET_IDS)}
SPLIT_MAP = {"train": "train", "validation": "val"}
SENSOR_FIELDS = ("temperature", "humidity", "vibration", "acceleration", "noise")


def label_policy() -> dict[str, object]:
    return {
        "version": POLICY_VERSION,
        "class_mapping": {
            str(source): {"name": TARGET_IDS[source], "yolo": index, "coco": index + 1}
            for source, index in CLASS_MAP.items()
        },
        "normal_ids": sorted(NORMAL_IDS),
        "split_mapping": dict(SPLIT_MAP),
        "exclude_issues": sorted(STRUCTURAL),
        "hold_issues": sorted(HOLD),
        "sensor_fields_checked": SENSOR_FIELDS,
    }
