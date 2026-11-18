# Licensing and data permissions

The repository currently declares no license for the project code. Obtain permission from the project owner before redistributing or relicensing it. Synthetic tests generate their own image, annotation, and sensor samples.

AI Hub dataset 71900 has its own access and usage conditions. Review your approval terms and the AI Hub policy before sharing archives, extracted samples, converted datasets, or models and reports that incorporate the data. Follow any attribution requirements for your release. The optional legacy `tools/aihubshell` is obtained separately from AI Hub and follows its upstream terms. [Dataset 71900](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=71900&topMenu=100) · [AI Hub data use policy](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do) · [AI Hub shell guide](https://www.aihub.or.kr/devsport/apishell/list.do)

Optional detector backends have their own licenses. Check the terms for the version you install and for any third-party weights you use.

| Component | Upstream license |
| --- | --- |
| PyTorch and TorchVision | [PyTorch BSD-style license](https://github.com/pytorch/pytorch/blob/main/LICENSE) and [TorchVision BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) |
| Ultralytics YOLO dependency | [AGPL-3.0 and Enterprise licensing options](https://www.ultralytics.com/license) |
| Transformers RT-DETR dependency | [Apache-2.0](https://github.com/huggingface/transformers/blob/v4.57.6/LICENSE) |

For other installed dependencies, retain the applicable license files and notices from their distributions.
