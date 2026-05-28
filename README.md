# Exp — Experiment Workspace Structure
#
# This directory is the clean, phase-by-phase workspace for the GalaxEye
# EO-SAR Binary Change Detection assignment.
#
# ── Phase Layout ──────────────────────────────────────────────────────
#
# Exp/
# ├── 00_data_audit.py          # Phase 0: Raw data understanding
# ├── 01_eda.py                 # Phase 1: Exploratory Data Analysis
# ├── 02_preprocessing.py       # Phase 2: Data cleaning & preprocessing
# ├── 03_feature_engineering.py  # Phase 3: Feature engineering (if needed)
# ├── 04_dataset.py             # Phase 4: PyTorch Dataset & DataLoader
# ├── 05_model.py               # Phase 5: Model architecture
# ├── 06_losses.py              # Phase 6: Loss functions
# ├── 07_train.py               # Phase 7: Training loop
# ├── 08_evaluate.py            # Phase 8: Evaluation & inference
# ├── 09_visualize.py           # Phase 9: Visualization & report figures
# │
# ├── configs/                  # All experiment configurations
# │   └── baseline.yaml
# │
# ├── reports/                  # EDA reports, audit JSONs, figures
# │   ├── figures/              # Saved plots
# │   └── data_audit_report.json
# │
# ├── checkpoints/              # Model weights
# │
# ├── logs/                     # Training logs, TensorBoard
# │
# └── notebooks/                # Optional Jupyter notebooks
