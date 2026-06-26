# PUMS Project

This repository contains a series of machine learning experiments on the PUMS system described in:

**“Proprioception with high spatiotemporal accuracy from unstructured multichannel sensing in soft robots.”**

Experiments are performed on:

- PUMS finger (surface estimation, surface classification, tool estimation)
- PUMS gripper (object recognition, width prediction)

---

# Experiments

## PUMS Finger

### 1. Surface Estimation

Estimation of the surface geometry pressed on a soft finger.

- A) training set reduction
- B) sensor reduction

---

### 2. Surface Classification

Classification of surfaces used in the surface estimation task. 35 classes and 4 contact positions

---

### 3. Tool Estimation (Tool position and force estimation in paper)

Estimation of tool position and applied force at the fingertip.

- A) training set reduction
- B) sensor reduction

---

## PUMS Gripper

### 1. Object Recognition (Object identification in paper)

Identification of grasped objects. 9 classes (8 objects + empty grasp)

- A) training set reduction
- B) sensor reduction
- C) finger flipping experiments

---

### 2. Width Prediction (Object length estimation in paper)

Prediction of object width under varying grasp angles.

- A) training set reduction
- B) sensor reduction
- C) finger flipping experiments

---

## Principal Component Analysis (PCA)

PCA is applied across datasets to analyze input structure.

- A) PCA on input matrices for each dataset
- B) removal of the last N principal components (surface estimation dataset)

---

# Installation

## 1. Clone repository

```bash
git clone <repository-url>
cd PUMS_code
```

## 2. Clone repository

```bash
python -m venv sr_env
source sr_env/bin/activate # on Linux/macOS and sr_env\Scripts\activate on Windows
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Prepare data and folders

- **CADs/**: 3D printed parts and molds (.stl)
- **experiments/**: ML pipelines (PCA, surface estimation, classification)
- **figures/**: Generated figures for publication
- **PCBs_pdf/**: PCB design exports (schematics + layers)
- **nogit_dataset/**: External datasets (Zenodo link: https://zenodo.org/uploads/20838108)
- **nogit_NN/**: Trained models (not versioned; create the folder)
- **results/**: Raw experiment outputs used for plots
