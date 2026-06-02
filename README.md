# DEC-Net

This repository is the official implementation of the paper:

**DEC-Net: Domain-Aware Expert Collaborative Network for Multi-Domain Medical Image Segmentation**

> ⚠️ This paper is currently under review.  More details will be updated upon acceptance.

## Datasets

DEC-Net is evaluated on multi-domain medical image segmentation benchmarks:

**Polyp Segmentation:**
- CVC-ClinicDB
- CVC-ColonDB
- ETIS

**Multi-Domain (Skin, Fundus, Thyroid):**
- ISIC 2017
- REFUGE2
- TN3K

## Project Structure

```
DEC-Net/
├── configs/            # Training configuration
│   └── config_setting.py
├── dataset/            # Dataset directory
├── dataprepare/        # Data preparation scripts
├── engine/             # Training and evaluation engine
│   ├── engine.py
│   ├── evaluate.py
│   └── test.py
├── models/             # Model architecture
│   ├── DEC_Net.py
│   ├── DAMoE/
│   └── bridge/
├── results/            # Output directory
├── ALRA.py
├── loader.py           # Data loader
├── train.py            # Training script
├── test.py             # Testing script
└── utils.py            # Utility functions
```

## Usage

### Requirements

- Python 3.x
- PyTorch
- torchvision

### Training

```bash
python train.py
```

Modify `configs/config_setting.py` to switch between different dataset configurations.

### Testing

```bash
python test.py
```

## License

This project is for academic research only.
