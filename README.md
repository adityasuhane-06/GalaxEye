# Building-Guided EO-SAR Change Detection

**Author:** Aditya Suhane  
**Status:** Completed  
**Assignment:** GalaxEye — Satellite AI Research Intern  

This repository contains the finalized code and research for the GalaxEye EO-SAR Binary Change Detection technical assignment. The goal is to detect structural damage to buildings following natural disasters using pre-event optical (EO) imagery and post-event synthetic aperture radar (SAR) imagery.

---

## 🚀 Key Innovations

### 1. The Extreme Class Imbalance Problem
A comprehensive data audit across 2.9 billion pixels revealed a massive **98.4% background class imbalance**. A naive deep learning model trained on this dataset learns to predict "no change" everywhere, achieving 98.4% accuracy with an Intersection over Union (IoU) of exactly 0. 

### 2. Building-Guided Architecture
To solve this, this repository implements a **Two-Stage Building-Guided Architecture**, heavily inspired by the 1st-place solution of the IEEE GRSS DFC 2025 competition:
*   **Stage 1 (Building Extractor):** A U-Net that processes only the pre-event EO image to extract robust building footprints.
*   **Stage 2 (Damage Classifier):** A late-fusion model that processes both EO and SAR features to predict structural damage.
*   **Final Prediction:** Damage predictions are strictly gated by the building footprint probability mask.

### 3. Masked Loss Functions
To prevent the 85% pure background noise from drowning out the damage signal during training, a **Masked Binary Cross-Entropy Loss** was implemented. The loss function zeroes out gradients for all non-building pixels, forcing the neural network to focus exclusively on structural damage.

---

## 📊 Results & Domain Shift Analysis

| Metric | Validation (Seen Events) | Test (Unseen Events) |
| :--- | :---: | :---: |
| **IoU** | 0.6486 | 0.3032 |
| **F1-Score** | 0.7869 | 0.4653 |

The model achieves an outstanding **0.86 IoU in dense urban environments (Scene 06)**. However, cross-event domain shift remains a challenge: the model struggles to generalize to unseen disaster domains, such as coastal earthquakes (Scene 10), where SAR radar speckle acts unpredictably and damage is invisible from an aerial optical view. 

*For a deep dive into the error analysis, read the [Technical Report](Technical_Report.md).*

---

## 📂 Repository Structure

```text
.
├── 00_data_audit.py        # Data audit & class imbalance analysis
├── 01_eda.py               # Exploratory Data Analysis & stats extraction
├── 04_dataset.py           # Custom PyTorch Dataset with Smart 10:1 Oversampling
├── 05_model.py             # Building-Guided Architecture definition
├── 06_losses.py            # Custom Masked BCE & Masked Dice losses
├── 07_train.py             # Mixed-precision training loop
├── run_eval.py             # Evaluation & Inference script
├── test.py                 # Secondary evaluation script
├── Technical_Report.md     # Full 20+ page research report & visual proofs
└── reports/figures/        # Data audit charts & inference visualisations
```

---

## 🛠️ Quick Start & Reproducibility

### 1. Model Weights
The best fully-trained model weights (Epoch 79, Val IoU=0.68) can be downloaded via Google Drive (Link provided in `Technical_Report.md` section 4.4).
Place the downloaded weights file as `best.pth` in the root directory.

### 2. Running Evaluation Locally
To evaluate the model on the validation or test splits and generate metrics/visualizations, simply run:

```bash
python run_eval.py
```
*(Ensure `best.pth` is in the root directory before running).*

### 3. Reading the Report
For a complete understanding of the methodology, architecture diagrams, training progressions, and visual evidence, please open **[`Technical_Report.md`](Technical_Report.md)** in any standard Markdown viewer.
