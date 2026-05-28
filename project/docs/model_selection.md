# Model Selection

## Baseline Ladder

The project deliberately separates model choices into a ladder:

1. Classical thresholding or random forest on EO/SAR differences: useful only as sanity baselines.
2. Plain U-Net: simple but weak for EO/SAR domain mismatch.
3. Siamese U-Net: useful for same-modality change detection, risky for EO/SAR if weights are shared.
4. Late-fusion EO/SAR U-Net: better because each modality has its own encoder.
5. Gated-difference EO/SAR U-Net: stronger because it learns when EO/SAR differences are meaningful.
6. Building-guided gated-difference U-Net: selected model.
7. Transformer/ChangeFormer-style model: future work if compute allows.

## Selected Model

The selected model has:

- EO ResNet34 encoder,
- SAR ResNet18 encoder,
- learned gated difference fusion at multiple scales,
- EO-only building decoder,
- change decoder guided by building features and building probability,
- three supervised heads:
  - binary change,
  - 4-class semantic damage,
  - building mask.

## Why This Is Appropriate

The model reflects three dataset facts:

1. EO and SAR are different modalities, so separate encoders are safer than shared weights.
2. Change occurs mostly on buildings, so a building prior reduces false positives.
3. Damaged/destroyed pixels are rare, so binary-only learning is unstable.

## Main Risks

- Building guidance can suppress true change outside buildings. This is acceptable because the assignment labels building damage.
- Too much positive oversampling can reduce precision. The config balances positive, intact, and texture hard-negative crops.
- Threshold chosen on validation may not transfer perfectly to test. The report should include threshold sensitivity.
