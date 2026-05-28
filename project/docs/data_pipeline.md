# Data Cleaning, Preprocessing, and EDA

## Data Validation

Before training, validate:

- every sample has matching `pre-event`, `post-event`, and `target` TIFF files,
- pre/post/mask shapes match,
- EO has 3 channels,
- SAR has 1 channel,
- target labels are only `{0, 1, 2, 3}`,
- filenames parse into scene IDs.

Run:

```bash
python validate_data.py --data_root ../data --output outputs/data_audit.json
```

## Label Engineering

Raw target:

```text
0 background
1 intact
2 damaged
3 destroyed
```

Derived targets:

```text
binary_change = target >= 2
building      = target >= 1
multiclass    = target
```

This is not external feature engineering; it is a legal remapping of labels already provided.

## Image Preprocessing

EO:

- convert to float `[0, 1]`,
- ImageNet normalization,
- optional brightness/contrast jitter,
- optional grayscale augmentation,
- optional RGB channel shuffle.

SAR:

- convert to float `[0, 1]`,
- normalize with mean `0.5`, std `0.5`,
- multiplicative speckle augmentation,
- small SAR-only shifts to simulate imperfect EO/SAR registration.

## Crop Sampling

Training uses crops because full 1024x1024 images are expensive and change pixels are sparse.

Sampling types:

- positive damage crops,
- intact-building hard negatives,
- high-texture SAR no-change hard negatives,
- random crops.

This helps prevent two common failures:

- predicting every building as damaged,
- predicting every SAR texture edge as change.

## EDA Outputs

The EDA script writes:

- global class counts,
- binary change fraction,
- building fraction,
- per-scene counts,
- per-channel statistics,
- shape/dtype summary,
- CSV/JSON summaries,
- optional plots if matplotlib is available.

Run:

```bash
python eda.py --data_root ../data --output_dir outputs/eda
```
