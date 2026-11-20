# Robustness–Efficiency Benchmarking of Lightweight CNNs for Waste Classification Under Low-Quality Camera Conditions

Official code for the paper **"Robustness–Efficiency Benchmarking of Lightweight CNNs for Waste Classification Under Low-Quality Camera Conditions"** by [Author Name], [University Name].

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

Automatic waste-sorting systems often rely on low-cost cameras that produce blurred, noisy, dark or heavily compressed images. This repository benchmarks four ImageNet-pretrained lightweight CNNs on the six-class TrashNet dataset, reporting **accuracy, model size, CPU latency and robustness** together:

- MobileNetV3-Small
- ShuffleNetV2 (1.0x)
- EfficientNet-B0
- SqueezeNet 1.1

Each model is evaluated on clean images and under **10 synthetic corruptions at 3 severity levels**. We also test whether **corruption-aware augmentation** trained on 5 *seen* corruptions transfers to 5 *unseen* corruptions.

## Key results

Mean ± SD over 3 seeds (accuracy, %). Seen/unseen = mean over 15 conditions.

| Model | Training | Clean | Seen corruptions | Unseen corruptions |
|---|---|---|---|---|
| MobileNetV3-S | Standard | 92.62 ± 1.26 | 72.16 ± 0.68 | 69.46 ± 0.34 |
| | Corruption-aware | 90.25 ± 2.37 | 84.37 ± 1.64 | 74.71 ± 1.45 |
| ShuffleNetV2 | Standard | 90.18 ± 0.57 | 63.48 ± 1.79 | 61.08 ± 1.42 |
| | Corruption-aware | 86.76 ± 1.89 | 80.07 ± 1.12 | 68.64 ± 1.04 |
| EfficientNet-B0 | Standard | 94.27 ± 0.68 | 70.83 ± 1.72 | 74.91 ± 3.98 |
| | Corruption-aware | 94.14 ± 0.80 | 88.59 ± 1.23 | 79.54 ± 0.20 |
| SqueezeNet 1.1 | Standard | 84.85 ± 1.99 | 50.83 ± 1.91 | 46.92 ± 2.89 |
| | Corruption-aware | 79.38 ± 1.48 | 71.98 ± 1.49 | 53.77 ± 1.62 |

Efficiency (representative run, seed 42):

| Model | Params (M) | Size (MB) | CPU latency (ms) |
|---|---|---|---|
| MobileNetV3-S | 1.52 | 6.10 | 11.8 |
| ShuffleNetV2 | 1.26 | 5.04 | 23.1 |
| EfficientNet-B0 | 4.02 | 16.06 | 34.1 |
| SqueezeNet 1.1 | 0.73 | 2.90 | 19.0 |

Main findings:

1. High clean accuracy does not prevent corruption-specific failure: EfficientNet-B0 drops to chance level (16.9%) under strong Gaussian noise.
2. Corruption-aware augmentation gives +12.2 to +21.2 points on seen corruptions but only +4.6 to +7.6 on unseen ones.
3. Gains are limited to related corruption families (motion blur, salt-and-pepper noise); there is no consistent gain for haze, contrast or exposure.

> **Caveat:** TrashNet may contain near-duplicate photographs, and no de-duplication was applied. Absolute accuracies may be optimistic; relative comparisons are the focus. See the paper's Limitations section.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── .gitignore
├── notebooks/            # Kaggle/Colab notebook(s) used for the experiments
├── src/                  # (optional) scripts: training, corruptions, evaluation, gradcam
├── results/              # CSV/JSON result tables
├── figures/              # figures used in the paper
└── data/                 
```

> Edit this tree to match your actual files.

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with: Python [version], PyTorch [version], torchvision [version], scikit-learn [version], SciPy [version].

## Dataset

The dataset is **not** redistributed here. Download the Garbage Classification (TrashNet) data from Kaggle:
<https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification>

Original TrashNet source: <https://github.com/garythung/trashnet>

Place the images so that the layout is:

```
data/
└── Garbage classification/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

Split: stratified 80/20 train+val/test with `random_state=42`, then 10% of train+val for validation (1,818 / 203 / 506 images).

## Usage

> Replace the commands below with those that match your code. If everything lives in a notebook, describe the order in which to run the cells.

```bash
# 1. Train (standard and corruption-aware), seeds 0, 1, 2
python src/train.py --model efficientnet_b0 --train_mode standard --seed 0
python src/train.py --model efficientnet_b0 --train_mode corruption_aware --seed 0

# 2. Evaluate on clean + 30 corruption conditions
python src/evaluate.py --model efficientnet_b0 --checkpoint checkpoints/efficientnet_b0_std_seed0.pt

# 3. Statistical analysis (bootstrap CIs, McNemar)
python src/stats.py

# 4. Grad-CAM
python src/gradcam.py --model efficientnet_b0
```

## Experimental setup

| Setting | Value |
|---|---|
| Input / batch size | 224×224 / 32 |
| Optimiser | AdamW, lr 5×10⁻⁴, weight decay 10⁻⁴ |
| Schedule | Cosine annealing (per iteration) |
| Loss | Cross-entropy, label smoothing 0.05 |
| Epochs | 8 (final models), 4 (CV folds) |
| Augmentation | Flip, rotation ±15°, colour jitter 0.2 |
| Seeds | 0, 1, 2 (multi-seed); 42 (representative run) |

### Corruptions

| ID | Corruption | Severity 1 / 2 / 3 |
|---|---|---|
| S1 | Gaussian blur | σ = 1 / 2 / 4 px |
| S2 | Gaussian noise | σ = 10 / 25 / 50 |
| S3 | Low light | brightness 0.6 / 0.35 / 0.15 |
| S4 | JPEG compression | quality 30 / 15 / 5 |
| S5 | Low resolution | scale 2 / 4 / 8 |
| U1 | Motion blur | kernel 7 / 15 / 25 px |
| U2 | Low contrast | factor 0.5 / 0.3 / 0.15 |
| U3 | Salt-and-pepper | fraction 0.02 / 0.06 / 0.12 |
| U4 | Haze | weight 0.3 / 0.5 / 0.7 |
| U5 | Overexposure | factor 1.6 / 2.2 / 3.0 |

S = seen during training augmentation, U = unseen (test only). Corrupted test images are generated once with a fixed seed.

## Reproducibility

- Fixed seeds: 0, 1, 2 (and 42 for the representative run).
- Corrupted test set generated once with a fixed random seed.
- GPU: Kaggle notebook GPU. CPU latency measured on a Kaggle CPU (batch 1, 2 threads, 50 runs after 10 warm-up). Latency is indicative only.
- Small numerical differences may occur across hardware and library versions.

## Citation

If you use this code, please cite:

```bibtex
@article{yourname2026waste,
  title   = {Robustness--Efficiency Benchmarking of Lightweight CNNs for Waste Classification Under Low-Quality Camera Conditions},
  author  = {[Author Name]},
  journal = {[Journal / arXiv]},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE). The TrashNet dataset has its own license; please check the original repository.

## Acknowledgments

Thanks to G. Thung and M. Yang for creating TrashNet, the Kaggle user who hosts the dataset copy, and Kaggle for free GPU/CPU resources.

## Contact

[Rayhan Kobir] · [rayhan.kobir.shah@gmail.com] · [North Bengal International University]
