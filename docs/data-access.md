# AI Hub SMT data access

Use your AI Hub account to request access to dataset **71900** and wait for approval. Follow AI Hub's API guide to issue a key for shell downloads. The dataset page lists eligibility and release history. This pipeline's packaged selection expects JPG images, JSON labels, and XLSX sensors for its selected Train and Validation folders. Verify the file keys and archive layout against your approved release. [Dataset 71900](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=71900&topMenu=100) · [AI Hub shell and API guide](https://www.aihub.or.kr/devsport/apishell/list.do)

Put `AIHUB_API_KEY` in the process environment or copy `.env.example` to the untracked workspace `.env` and fill in the value locally. Keep it out of command arguments, YAML, shell history, dataset paths, Git, and files you share. Environment values override `.env` values. Use the [AI Hub data use policy](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do) and [licensing notes](licenses.md) to check sharing permissions for source data, derived datasets, and published results.

## Reuse existing files

- **Extracted data**. Use `ict setup --non-interactive --dataset-source skip --data-path data/extracted` and `ict run` to read the tree in place. The English layout uses immediate child folders such as `train_images_soldering_defect_cold_solder` and `validation_annotations_soldering_defect_cold_solder`, with a matching stem across JPG, JSON, and XLSX. For an already downloaded legacy archive set, prefer the extractor to produce this layout in a new directory. The legacy path migration script previews renames by default.
- **Raw archives**. Place approved local ZIPs in `data/raw` and run `ict data status`, `ict data verify`, and then `ict data extract`. Pick a new extraction destination via `ict data --data-path "path/to/new-extraction" extract` if the default already contains other files. For ingestion inside a resumable run, write a profile with `dataset_source: aihub` and `data_path: data/extracted`, then use `ict run --config profile.yaml --run-id smt`. Complete local archives or verified download checkpoints are reusable without a key, network, or transfer approval.
- **Processed detections**. Reuse the directory containing `yolo/` and `coco/` with `ict train --dataset` and `ict evaluate --dataset`. To reuse a pipeline conversion, run `ict resume --run-id <id>` with the same profile. The runner verifies the saved artifacts and prints the conversion path. A run's published archive snapshot is its source for extraction. Start a new run or use `--force` to reselect raw archives after replacing or adding archives.

## Acquire missing archives

```text
ict setup --non-interactive --dataset-source aihub --data-path data/extracted
ict data download
```

The download preview lists missing splits, storage caps, and the raw destination. Review free space on both the archive and artifact filesystems. Extraction and preprocessing need additional space for their own copies. After approving the transfer, use `ict run --run-id smt --approve-download` to acquire and prepare the data, or `ict data download --approve-download` followed by `ict data extract`. Linux, WSL2, and macOS use curl. The Windows downloader needs an existing WSL distribution with curl. `ICT_WSL_DISTRIBUTION` selects a distribution when needed.
