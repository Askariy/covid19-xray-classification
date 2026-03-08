# COVID-19 Chest X-Ray Classification

Multi-class classification of chest X-ray images for **COVID-19**, **Pneumonia**, and **Normal** using deep learning. This project compares likelihood (BCE) and **Focal Loss** with pretrained **VGG16** and **ResNet-18**, including hyperparameter sweeps and test-set evaluation with CSV export.

---

## Dataset

This project uses **COVIDx-CXR**:

| Source | Description | Link |
|--------|-------------|------|
| **COVIDx-CXR** | ~3,200 images, 3 classes (COVID, Pneumonia, Normal), train/test split | [Figshare](https://figshare.com/articles/dataset/COVID-Pneumonia_Detection_Expert-Selected_X-ray_Dataset_COVIDx-CXR_/25917340) |

Arrange the data under a single root with three folders: **Train**, **Validation**, and **Test**, each with class subfolders (e.g. `COVID`, `Pneumonia`, `Normal`). Use the dataset’s train/test split and create Validation from a subset of the training set if needed.

*Related sources (not used in this repo):* [COVID-19 Chest X-Ray Repository](https://figshare.com/articles/dataset/COVID-19_Chest_X-Ray_Image_Repository/12580328) (Figshare, COVID-positive only), [Chest XR COVID-19 Grand Challenge](https://cxr-covid19.grand-challenge.org/Dataset/) (20k+ images, 3 classes).

---

## Setup

- **Python**: 3.x  
- **PyTorch** (with CUDA if available)  
- **torchvision**, **NumPy**, **scikit-learn**, **matplotlib**, **seaborn**, **tqdm**, **PIL**

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn tqdm pillow
```

The notebook is set up for **Google Colab** (Drive mount for data and results). For local runs, set `DATA_DIR` and `FIG_OUTPUT_DIR` to your paths and ensure `Train/`, `Validation/`, and `Test/` exist under `DATA_DIR`.

---

## Project structure

```
covid19-xray-classification/
├── README.md
└── Code File/
    └── covid19_classification_focal_loss.ipynb
```

---

## Models and training

- **Architectures**: Pretrained **VGG16** and **ResNet-18** with custom classification heads (3-class output).
- **Losses**:
  - **BCE (likelihood)**: `BCEWithLogitsLoss` with one-hot multi-label targets.
  - **Focal Loss**: Custom `FocalLoss(alpha, gamma)` to down-weight easy examples and handle class imbalance.
- **Training**: SGD (momentum 0.9), 10 epochs; multiple (α, γ, LR) configurations explored for both backbones.
- **Augmentation**: Random rotation, resize, RandomResizedCrop(224), horizontal flip; ImageNet normalization.

Best checkpoints (from the notebook):

- **VGG16 Focal**: `vgg16_focal_loss_exp5.pth` (α=1.5, γ=0.5, LR=0.001)
- **ResNet-18 Focal**: `res18_focal_loss_exp3.pth` (α=2, γ=1, LR=0.01)

---

## How to run

1. **Colab**: Upload the notebook, mount Drive, set `DATA_DIR` to your data root (e.g. `covid_xray_data` with `Train/`, `Validation/`, `Test/`). Run cells in order.
2. **Local**: Open the notebook or run with Jupyter; install dependencies and point `DATA_DIR`/`FIG_OUTPUT_DIR` to your machine.

Training is executed by running the corresponding notebook cells (no separate `train.py`). Each “Training Loop” section trains one configuration and saves a `.pth` checkpoint.

---

## Outputs

- **Figures**: Accuracy/loss curves and train/validation confusion matrices (saved via `save_fig()` to `FIG_OUTPUT_DIR`).
- **Test predictions**: Sections “Testing ResNet-18 Focal Loss” and “Testing VGG16 Focal Loss” run the best models on the test loader and write per-image predictions to **resnet18_focal_loss.csv** and **vgg16_focal_loss.csv** (filename + predicted class columns).

---
