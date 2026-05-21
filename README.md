# DSAF-Net: Dual-Stream Attention Fusion Network for Cardiovascular Risk Profiling via Fundus Imaging

> **Bio-Inspired Dual-Scale Attention Fusion Network for Explainable Cardiovascular Risk Profiling via Fundus Imaging**
> Deka, Debnath, Chatterjee, Kothapalli, Navatejareddy — *IJIES*

---

## Overview

DSAF-Net is a multi-task deep learning framework for **non-invasive cardiovascular disease (CVD) risk assessment** using retinal fundus images. It simultaneously performs:

1. **Vessel segmentation** — binary segmentation of the retinal microvasculature.
2. **CVD risk classification** — 3-class risk stratification (Low / Medium / High).

### Key innovations

| Component | Description |
|-----------|-------------|
| **Spatial stream** | ResUNet encoder with CBAM attention (channel + spatial) |
| **Spectral stream** | 3-level Daubechies-4 DWT decomposition + CNN |
| **Cross-stream attention** | Bidirectional multi-head attention: `As→w` and `Aw→s` |
| **Stream consistency loss** | `max(0, cos(Fs, Fw) − τ)` enforces complementary (non-redundant) features |
| **Multi-task loss** | `0.4·Lseg + 0.4·Lcls + 0.2·Lcons` |

### Results (DRIVE dataset)

| Method | Dice ↑ | CVD Accuracy ↑ |
|--------|--------|----------------|
| U-Net | 0.8123 | 82.31% |
| ResUNet | 0.8489 | 84.56% |
| TransUNet | 0.8556 | 88.94% |
| **DSAF-Net** | **0.8823** | **92.34%** |

---

## Repository Structure

```
dsafnet/
├── models/
│   ├── dsafnet.py       # Full DSAF-Net architecture
│   └── losses.py        # Segmentation, classification, and consistency losses
├── data/
│   └── dataset.py       # FundusDataset + CLAHE preprocessing + proxy CVD labelling
├── utils/
│   └── metrics.py       # Dice, IoU, AUC, confusion matrix, etc.
├── scripts/
│   ├── train.py         # Training loop with early stopping
│   └── evaluate.py      # Inference & evaluation on a test split
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/dsafnet.git
cd dsafnet
pip install -r requirements.txt
```

---

## Data Preparation

The code expects each split in the following layout:

```
dataset/
  train/
    images/   *.png  (RGB fundus images)
    masks/    *.png  (binary vessel ground truth)
  val/
    images/
    masks/
  test/
    images/
    masks/
```

### Supported datasets

| Dataset | Images | Resolution |
|---------|--------|------------|
| [DRIVE](https://drive.grand-challenge.org/) | 40 | 584×565 |
| [STARE](http://cecas.clemson.edu/~ahoover/stare/) | 20 | 700×605 |
| [CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/) | 28 | 999×960 |

### CVD risk labels

Ground-truth CVD labels are **not** available in these datasets.  
The code uses a **proxy labelling strategy** (as in the paper): quantitative vascular biomarkers (vessel density, caliber, tortuosity, branching density) are extracted from the segmentation masks and thresholded at ±1 SD of the population mean to assign Low / Medium / High risk.

To use your own labels, provide a CSV:

```csv
filename,label
image001.png,0
image002.png,2
...
```

---

## Training

```bash
python scripts/train.py \
    --data_root /path/to/dataset \
    --save_dir checkpoints/ \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --amp               # optional: mixed precision
```

Key hyperparameters (paper defaults):

| Parameter | Value |
|-----------|-------|
| Image size | 512×512 |
| Batch size | 4 |
| Optimizer | Adam (lr=1e-4, wd=1e-5) |
| Epochs | 100 |
| Early stopping | patience=15 |
| Wavelet | Daubechies-4, 3 levels |
| Loss weights λ1,λ2,λ3 | 0.4, 0.4, 0.2 |
| Consistency threshold τ | 0.3 |

---

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_root /path/to/dataset/test \
    --output_dir results/ \
    --save_masks        # optional: save predicted vessel masks
```

---

## Architecture Details

### Mathematical formulation

```
Spatial features:    Fs = ResUNet-CBAM(I; θs)

Spectral features:   {ILL, ILH, IHL, IHH} = DWT3(I)
                     Fw = CNN_spectral([ILL, ILH, IHL, IHH]; θw)

Cross-attention:     As→w = softmax(Qs Kw^T / √dk) Vw
                     Aw→s = softmax(Qw Ks^T / √dk) Vs

Fused features:      Ffused = concat(Fs + As→w, Fw + Aw→s)

Predictions:         M̂  = Decoder_seg(Ffused)       [vessel mask]
                     ŷ  = Classifier_CVD(GAP(Ffused)) [CVD risk]

Loss:                L_seg  = 0.6·Focal + 0.4·Dice
                     L_cls  = CrossEntropy
                     L_cons = max(0, cos(Fs, Fw) − 0.3)
                     L_total = 0.4·L_seg + 0.4·L_cls + 0.2·L_cons
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{deka2024dsafnet,
  title   = {Bio-Inspired Dual-Scale Attention Fusion Network for Explainable
             Cardiovascular Risk Profiling via Fundus Imaging},
  author  = {Deka, Bhupesh and Debnath, Dipwanita and Chatterjee, Sayanti and
             Kothapalli, Pavan Kumar Varma and Navatejareddy, Ramireddy},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {2024}
}
```

---

## License

This implementation is released under the MIT License.
