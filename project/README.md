# GalaxEye EO-SAR Binary Change Detection

Clean research project for the GalaxEye Satellite AI Research Intern assignment.

## Problem

Given a co-registered pre-event electro-optical image and post-event SAR image, predict a binary pixel-level change mask:

- `0`: no change, from original classes `background` and `intact`
- `1`: change, from original classes `damaged` and `destroyed`

The dataset also contains useful intermediate semantics:

- `building = intact + damaged + destroyed`
- `change = damaged + destroyed`

This project uses that structure instead of treating the task as plain binary segmentation.

## Research Direction

The main model is inspired by building-guided cross-modal damage mapping:

1. Learn a building prior from the pre-event optical image.
2. Fuse EO and SAR features with gated feature differences.
3. Predict binary change, 4-class damage labels, and building mask jointly.
4. Use the building prior at inference to suppress non-building false positives.

This follows the practical lesson from BGPLL: cross-modal EO/SAR damage mapping improves when the model knows where buildings are before deciding whether they changed.

## Expected Data Layout

Place the assignment data as:

```text
data/
  train/train/
    pre-event/*.tif
    post-event/*.tif
    target/*.tif
  val/val/
    pre-event/*.tif
    post-event/*.tif
    target/*.tif
  test/test/
    pre-event/*.tif
    post-event/*.tif
    target/*.tif
```

The scripts are forgiving and also accept paths directly to the split folder containing `pre-event`, `post-event`, and `target`.

## Setup

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Kaggle, install only missing packages if needed; PyTorch and CUDA are already present.

## Data Audit and EDA

Run this before training:

```bash
python validate_data.py --data_root ../data --output outputs/data_audit.json
python eda.py --data_root ../data --output_dir outputs/eda
```

This checks triplet matching, image shapes, label values, class imbalance, scene distribution, and per-modality statistics.

## Cross-Event Experiment

Use scenes `01-06` for training and `07-08` for validation. This is not the official validation result; it is a stress test for domain shift.

```bash
python train.py \
  --config configs/building_guided_cross_event.yaml \
  --train_dir ../data/train/train \
  --val_dir ../data/val/val \
  --device cuda
```

Evaluate the best checkpoint:

```bash
python eval.py \
  --config configs/building_guided_cross_event.yaml \
  --data_path ../data/val/val \
  --weights outputs/checkpoints_building_guided_cross_event/best.pth \
  --output outputs/metrics/cross_event_val.json \
  --full_image \
  --sweep_thresholds \
  --binary_source building_guided \
  --device cuda
```

## Final Training

After cross-event sanity checks, train on the full provided training split and evaluate on validation/test.

```bash
python train.py \
  --config configs/building_guided_final.yaml \
  --train_dir ../data/train/train \
  --val_dir ../data/val/val \
  --device cuda
```

```bash
python eval.py \
  --config configs/building_guided_final.yaml \
  --data_path ../data/test/test \
  --weights outputs/checkpoints_building_guided_final/best.pth \
  --output outputs/metrics/test_metrics.json \
  --threshold 0.4 \
  --full_image \
  --binary_source building_guided \
  --visualize \
  --device cuda
```

## Main Files

- `docs/problem_definition.md`: assignment interpretation
- `docs/literature_review.md`: papers and how each one is used
- `docs/data_pipeline.md`: cleaning, preprocessing, EDA, feature construction
- `src/galaxeye_research_cd/data.py`: dataset, remapping, sampling, augmentations
- `src/galaxeye_research_cd/models.py`: building-guided EO/SAR model
- `src/galaxeye_research_cd/losses.py`: binary, 4-class, building multi-task loss
- `train.py`: training loop
- `eval.py`: full-image tiled evaluation and threshold sweep

## Core References

- Li et al., Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping, 2025: https://arxiv.org/abs/2505.04941
- Chen et al., BRIGHT: a globally distributed multimodal building damage assessment dataset, ESSD 2025: https://essd.copernicus.org/articles/17/6217/2025/
- Bandara and Patel, ChangeFormer: A Transformer-Based Siamese Network for Change Detection, 2022: https://arxiv.org/abs/2204.00154
- Zheng et al., Neural Disaster Simulation for Transferable Building Damage Assessment, 2025: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5228282
