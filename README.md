# MHR-YOLO

Official source code for **Insulator Defect Detection via Sparse Mixture-of-Experts and Region Prototype Learning**.

Built on YOLO11n, **MHR-YOLO** targets high-accuracy insulator defect detection under heterogeneous defect morphologies, small-target regression sensitivity, and cluttered aerial inspection backgrounds with a controlled complexity budget (3.25M total parameters, 3.04M active inference parameters, and 9.3 GFLOPs).

## Architecture Overview

| Paper Module | Implementation | Description |
| :--- | :--- | :--- |
| **DP-MoE** | `ultralytics/nn/modules/ScaleAwareMoE_Block.py` (`DP_MoE`) | Dual-pooling (GAP + GMP) routed sparse mixture-of-experts at P3 and P4, combining an always-on 3×3 shared branch with four multi-kernel depthwise-separable experts (3×3, 5×5, 7×7, 3×3) under Top-2 conditional execution and cosine temperature annealing ($\tau: 1.5 \to 0.5$). |
| **DOHR** | `ultralytics/utils/loss.py` (`DOHRBboxLoss`) | Distance–overlap hybrid regression equally fusing Normalized Wasserstein Distance (NWD, $C_{\mathrm{nwd}}=12.8$) and CIoU alongside Distribution Focal Loss (DFL). |
| **CB-RPL** | `ultralytics/nn/modules/head.py` (`CBRPLDetect`) & `ultralytics/utils/loss.py` | Training-only class-balanced region prototype learning via $3\times3$ multi-scale grid sampling inside ground-truth boxes and stop-gradient EMA prototypes ($\alpha=0.95$, $T=0.1$). Bypassed completely at inference for zero runtime overhead. |

*(Legacy aliases `CSP_MoE_Block`, `ContrastiveDetect`, `NWDBboxLoss`, and `v8PGM3OptimizedDetectionLoss` are preserved internally for full checkpoint compatibility.)*

## Main Results

- **FCIDD Test Set (365 images, 899 instances):** `87.73%` $\mathrm{mAP}_{50}$ and `57.24%` $\mathrm{mAP}_{50:95}$ (+3.46 and +4.42 percentage points over YOLO11n; #1 among 12 compared detectors).
- **FCIDD Validation (3-seed mean ± SD):** `85.87 ± 0.39%` $\mathrm{mAP}_{50}$ and `54.24 ± 0.91%` $\mathrm{mAP}_{50:95}$ (+2.85 and +3.90 percentage points over YOLO11n).
- **CPMID Cross-Dataset Validation:** `74.87%` $\mathrm{mAP}_{50}$ and `52.20%` $\mathrm{mAP}_{50:95}$ (+1.31 and +1.47 percentage points over YOLO11n).

## Installation

Create a Python 3.10+ environment (experiments were run with Python 3.12, PyTorch 2.12, CUDA 13.0) and install PyTorch matching your CUDA toolkit, followed by the remaining dependencies:

```bash
pip install -r requirements.txt
```

The training and evaluation entry points automatically load the bundled modified `ultralytics` runtime in this repository.

## Datasets

1. **FCIDD** (4 classes: `flashover`, `break`, `drop`, `insulator`): available at [https://github.com/Insistc/dataset](https://github.com/Insistc/dataset). Partitioned into 2,921 training, 366 validation, and 365 test images.
2. **CPMID** (3 classes): 5,398 training, 1,542 validation, and 772 test images.

Copy `configs/data/fcidd.example.yaml` to a local YAML file and update `path` to point to your dataset directory in standard YOLO layout (`images/{train,val,test}` and `labels/{train,val,test}`).

## Training and Evaluation

Train MHR-YOLO from scratch on FCIDD (640×640, 200 epochs, batch size 32, SGD `lr0=0.01`, `dfl=1.5`):

```bash
python scripts/train.py --data /path/to/fcidd.yaml --model configs/models/mhr-yolo11n-fcidd.yaml --device 0 --seed 0
```

For multi-seed evaluation, repeat with `--seed 1` and `--seed 2`. For CPMID, pass `--model configs/models/mhr-yolo11n-cpmid.yaml` with your CPMID data YAML.

Evaluate a trained checkpoint on the validation or test split:

```bash
python scripts/evaluate.py --weights runs/train/mhr-yolo-seed0/weights/best.pt --data /path/to/fcidd.yaml --split test --device 0
```

## Repository Structure

```text
MHR-YOLO/
├── configs/
│   ├── data/fcidd.example.yaml
│   ├── models/
│   │   ├── mhr-yolo11n-fcidd.yaml
│   │   └── mhr-yolo11n-cpmid.yaml
│   └── train.yaml
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── ultralytics/          # Modified Ultralytics 8.4.21 runtime (DP-MoE, DOHR, CB-RPL)
├── requirements.txt
└── LICENSE
```

## License and Acknowledgements

This repository builds upon [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and is released under the GNU AGPL-3.0 license (see [LICENSE](LICENSE)). Citation information will be updated upon publication.
