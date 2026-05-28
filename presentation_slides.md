# Building-Guided EO-SAR Change Detection
**Visual Presentation & Data Proof**

---

## 1. The Core Problem: Extreme Class Imbalance

Before designing the architecture, a comprehensive data audit was run across the 2.9 billion pixels in the training set. The audit revealed a massive 62.7:1 imbalance.

![Class Imbalance Distribution](C:/Users/Lenovo/.gemini/antigravity-ide/brain/8255d1bc-ae5a-42c2-8e46-9ed8ad04c2af/figures/class_imbalance.png)

> [!IMPORTANT]
> **Key Finding:** 98.4% of all pixels are background. A standard segmentation model predicts "no change" everywhere and achieves 98.4% accuracy with 0 IoU. This physical constraint required a massive architectural rethink.

---

## 2. The Solution: Building-Guided Masking

Because damage *only* happens to buildings, the problem was split into two stages:
1. Extract building footprints from the optical (EO) image.
2. Evaluate damage using EO and SAR *only within those building footprints*.

![Building vs Background Statistics](C:/Users/Lenovo/.gemini/antigravity-ide/brain/8255d1bc-ae5a-42c2-8e46-9ed8ad04c2af/figures/building_vs_background.png)

By masking the loss function during training, the 85% pure background noise was silenced (zero gradient), forcing the model to focus purely on structural changes.

---

## 3. Training Success: Dense Urban Generalization (Scene 06)

The building-guided approach excelled in dense urban environments where structures were clearly visible. Scene 06 (Dense Urban) achieved an outstanding **IoU of 0.86**.

![Dense Urban Training Success](C:/Users/Lenovo/.gemini/antigravity-ide/brain/8255d1bc-ae5a-42c2-8e46-9ed8ad04c2af/figures/triplets_train_scene06.png)

> [!NOTE]
> *Left:* Pre-disaster Optical (EO)  
> *Middle:* Post-disaster Radar (SAR)  
> *Right:* Ground Truth Mask (Grey = No Change, Red = Damaged)

---

## 4. The Limitation: Cross-Event Domain Shift (Test Scene 10)

While the model generalized well to seen disaster types (Disaster Type 5, Disaster Type 1), it struggled severely on unseen disaster types. Test Scene 10 was a coastal Disaster Type 2, which the model had never encountered.

![Disaster Type 2 Domain Shift Failure](C:/Users/Lenovo/.gemini/antigravity-ide/brain/8255d1bc-ae5a-42c2-8e46-9ed8ad04c2af/figures/triplets_test_scene10.png)

> [!WARNING]
> **Why it failed (IoU 0.037):** 
> 1. Disaster Type 2 damage is often invisible from above (roofs look intact in optical imagery).
> 2. SAR radar over dense coastal cities creates massive speckle noise and shadows that the model had not learned to interpret.

*Note: Even the 1st place GRSS 2025 competition winner suffered massive test set drops (down to 9.7% IoU on the damaged class) due to this exact cross-event domain shift phenomenon.*

