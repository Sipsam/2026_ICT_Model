# External PCB COCO annotation guide

Place external PCB images in a separate directory and record each file under
`coco/images/external/`. Add labels to `instances_external.json` using the
canonical five-class mapping:

| category_id | name |
|---|---|
| 1 | cold_solder |
| 2 | misalignment |
| 3 | short |
| 4 | missing_component |
| 5 | tombstone |

Use absolute original-image pixel coordinates:

```json
{
  "image_id": 1,
  "bbox": [120.0, 80.0, 42.0, 36.0],
  "category_id": 1,
  "area": 1512.0,
  "iscrowd": 0
}
```

The `bbox` is `[x, y, width, height]`. Width and height must be positive and
all coordinates must remain inside the original image. Give every image and
annotation a unique integer ID. Keep `categories` unchanged. Do not add
fabricated external images or ground-truth boxes.

When the external annotation contains no images or annotations, the benchmark
reports it as unlabeled and does not emit AP, Precision, Recall, or IoU metrics.
