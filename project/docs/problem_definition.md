# Problem Definition

## Objective

The assignment is binary pixel-level change detection on paired EO/SAR imagery. Each sample contains:

- pre-event EO image: RGB, visually interpretable but weather/light sensitive
- post-event SAR image: single-channel radar, all-weather but noisy and structurally different from EO
- target mask: original 4-class semantic label

The required output is a binary mask:

```text
1 = changed
0 = unchanged
```

Mandatory label remapping:

| Original class | Original value | Binary value |
|---|---:|---:|
| Background | 0 | 0 |
| Intact | 1 | 0 |
| Damaged | 2 | 1 |
| Destroyed | 3 | 1 |

## Research Framing

This is not ordinary binary segmentation. The important hidden structure is:

```text
building = intact + damaged + destroyed
change   = damaged + destroyed
```

The model should learn:

1. where buildings are,
2. how EO and SAR disagree,
3. which disagreements correspond to disaster damage rather than modality noise.

## Main Difficulty

The visible validation split and visible test split may have different event distributions. This creates cross-event domain shift:

- different disaster type or severity,
- different urban morphology,
- different SAR texture/noise,
- different change pixel frequency,
- different ratio of damaged vs destroyed buildings.

High validation IoU with low test IoU usually means the model learned event-specific appearance rather than transferable damage cues.

## Evaluation

Metrics are computed only for the change class:

- IoU
- precision
- recall
- F1
- confusion matrix

Accuracy is not a primary metric because no-change pixels dominate.
