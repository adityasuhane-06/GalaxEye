# Literature Review and How It Is Used

## 1. Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping

Reference: Li et al., 2025, https://arxiv.org/abs/2505.04941

This is the closest paper to the assignment. It uses pre-disaster optical and post-disaster SAR images for building damage mapping with four labels: background, intact, damaged, and destroyed.

Key ideas:

- train building extraction from pre-event optical imagery,
- use building priors to guide damage prediction,
- use low-uncertainty pseudo-label refinement,
- use TTA and post-processing,
- report strong performance in the 2025 IEEE GRSS Data Fusion Contest.

How this project uses it:

- derive an auxiliary building target from the provided labels,
- predict building mask jointly with binary change and 4-class damage,
- multiply binary change probability by building probability at inference,
- sample intact buildings as hard negatives.

What is not used:

- pseudo-label training on the test set, because the assignment requires respecting the provided train/validation/test split.

## 2. BRIGHT Dataset Paper

Reference: Chen et al., ESSD 2025, https://essd.copernicus.org/articles/17/6217/2025/

BRIGHT describes a globally distributed multimodal building damage assessment dataset for all-weather disaster response. It emphasizes EO/SAR modality mismatch, event diversity, and cross-event generalization.

How this project uses it:

- validates the choice of event-heldout validation,
- motivates reporting domain shift explicitly,
- motivates EO/SAR-specific normalization and augmentations,
- motivates class-wise analysis beyond aggregate IoU.

## 3. ChangeFormer

Reference: Bandara and Patel, 2022, https://arxiv.org/abs/2204.00154

ChangeFormer uses a transformer-based Siamese architecture for remote sensing change detection. Its core lesson is that multi-scale context and long-range dependencies help change detection.

How this project uses it:

- informs the multi-scale encoder-decoder structure,
- motivates feature-level change reasoning rather than simple image differencing,
- listed as a future architecture upgrade if compute allows.

Why not the main implementation:

- standard ChangeFormer assumes more similar pre/post image statistics than EO/SAR,
- our current priority is building-guided domain robustness under limited compute.

## 4. ChangeMamba and UACD-Style Change Detection

BGPLL compares against ChangeMamba and UACD. These papers show that modern sequence/state-space and uncertainty-aware change models are strong baselines.

How this project uses them:

- uncertainty appears in our analysis and future work,
- the current model keeps a simpler uncertainty-free implementation for reproducibility,
- threshold sweep and component filtering are used as practical uncertainty-aware evaluation tools.

## 5. Neural Disaster Simulation

Reference: Zheng et al., 2025, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5228282

This paper proposes generating synthetic post-disaster images and labels to improve transfer to unseen disaster events.

How this project uses it:

- future work only,
- supports the argument that unseen-event transfer is the central challenge.

Why not used for training:

- external/generated data would violate the assignment rule requiring only the provided dataset.

## Final Methodological Choice

The project chooses a conservative, assignment-compliant version of BGPLL:

```text
provided labels only
building auxiliary target
binary change target
4-class auxiliary target
EO/SAR gated feature fusion
building-guided inference
```
