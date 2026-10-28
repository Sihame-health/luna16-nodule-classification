# Pulmonary Nodule Classification on CT Scans (LUNA16)

Master's project (Digital Engineering for Healthcare, ENSAM Casablanca, 2025–2026).
A proof of concept comparing three deep learning approaches to classify 3D CT patches as **nodule / non-nodule**. Not clinically validated.

## Overview

- **Data:** LUNA16 (derived from LIDC-IDRI), MetaImage `.mhd/.raw` CT scans read with SimpleITK. 242 scans processed from subsets 3–9 (601 scans available). Source used: [Kaggle LUNA16 dataset](https://www.kaggle.com/datasets/tessnimetteib/luna16-dataset). The data is not included in this repository.
- **Preprocessing:** Hounsfield unit conversion, clipping to [-1000, 400], Z-score normalization, world-to-voxel coordinate conversion, extraction of 32×48×48 3D patches.
- **Dataset:** 465 nodule patches and 968 non-nodule patches, balanced to 968/968 with horizontal-flip augmentation (1,936 patches), split 70/15/15 into train/validation/test.

## Models

| Model | Input | Approach |
|---|---|---|
| 3D CNN + Random Forest | 3D patch (32×48×48) | 3 Conv3D blocks as feature extractor (128-D), Random Forest classifier |
| U-Net + VGG16 | central 2D slice (48×48) | Transfer learning with a frozen ImageNet VGG16 encoder |
| YOLOv8 | central 2D slice (64×64) | Ultralytics YOLOv8, AdamW, 25 epochs |

## Results

Balanced test set of 291 patches (145 nodules, 146 non-nodules).

| Model | Accuracy | Recall | Specificity | AUC |
|---|---|---|---|---|
| 3D CNN + Random Forest | 91.07% | 95.86% | 86.30% | 0.9613 |
| U-Net + VGG16 | 92.10% | 99.31% | 84.93% | 0.9885 |
| YOLOv8 | 96.91% | 99.31% | 94.52% | 0.9929 |

## Scope and next steps

This is a patch-level classification study on a balanced dataset, meant to compare architectures rather than to estimate clinical performance. Planned improvements:

- Scanner-level (or LUNA16 subset-level) train/test split, with augmentation applied to the training set only
- Hard negatives from `candidates.csv` and FROC evaluation, the official LUNA16 metric
- End-to-end 3D models and explainability (Grad-CAM)
- Validation on an external dataset

## Repository structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── luna16_pipeline.ipynb   full pipeline (Kaggle, Tesla T4 GPU)
└── report/
    └── LUNA16_Report.pdf       full project report (25 pages)
```

## Usage

```bash
pip install -r requirements.txt
```

Download LUNA16 (link above) and run `notebooks/luna16_pipeline.ipynb`. Experiments were run on Kaggle (Python 3.12, TensorFlow/Keras, Ultralytics YOLOv8).

## Author

Siham AIT TALEB, Master's student in Digital Engineering for Healthcare, ENSAM Casablanca.
