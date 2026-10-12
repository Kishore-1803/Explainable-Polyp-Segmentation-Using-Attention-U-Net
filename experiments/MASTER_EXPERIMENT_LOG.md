# MASTER EXPERIMENT LOG
## Endoscopic Lesion Detection & Risk Grading System
### Attention U-Net on Kvasir-SEG + CVC-ClinicDB Cross-Dataset Evaluation

> **Last updated:** 2026-08-21  
> **Single source of truth for all experimental results. Do not edit results manually.**  
> All metrics verified against `results/*/metrics.json` and `results/cvc_threshold_analysis/threshold_comparison.json`.

---

## Project Overview

| Item | Detail |
|---|---|
| Task | Binary polyp segmentation (lesion vs. background) |
| Primary dataset | Kvasir-SEG (1000 images, Simula Research Lab) |
| External dataset | CVC-ClinicDB (612 images, Bernal et al. 2015, Olympus colonoscope) |
| Primary architecture | Attention U-Net (custom, 7,851,773 parameters) |
| Baseline architecture | Standard U-Net (same depth, no attention gates) |
| Loss function | Combined BCE + Dice Loss (equal weight) |
| Optimizer | Adam |
| Framework | PyTorch |
| Hardware | CUDA GPU |

---

## Dataset & Split Details

### Kvasir-SEG (Experiments 1–4)
| Split | Ratio | Count |
|---|---|---|
| Train | 80% | 800 images |
| Validation | 10% | 100 images |
| Test | 10% | 100 images |
| Random seed | 42 | — |

### CVC-ClinicDB (Experiments 5–7)
| Item | Detail |
|---|---|
| Total | 612 images (all used for evaluation) |
| Source | Kaggle: balraj98/cvcclinicdb |
| Resolution | 384×288 px (native) |
| Split used | None — full 612 as external test set |
| Training on CVC | **Never. Zero-shot evaluation only.** |

### Preprocessing (all experiments unless noted)
| Step | Detail |
|---|---|
| Resize | 256×256 (BILINEAR for images, NEAREST for masks) |
| Normalization | ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| Mask binarization | Threshold > 0.5 after /255 |
| Training augmentation | H-flip, V-flip, 90/180/270° rotation, ColorJitter (prob=0.5 each) |
| Inference augmentation | None |

---

## Experiment Catalogue

---

### EXP-01 — Attention U-Net Baseline (Kvasir-SEG, 30 Epochs)

| Field | Value |
|---|---|
| **Experiment type** | Training — primary baseline |
| **Date** | Review 1 (August 2026) |
| **Architecture** | Attention U-Net |
| **Checkpoint** | `checkpoints/review_1/review_1_best.pth` |
| **Checkpoint saved at** | Epoch 16 (best val Dice during training) |
| **Dataset (train)** | Kvasir-SEG |
| **Dataset (test)** | Kvasir-SEG test split (100 images) |
| **Dropout** | None (0.0) |
| **Epochs** | 30 |
| **Batch size** | 8 |
| **Learning rate** | 1e-4 |
| **Weight decay** | 1e-5 |
| **Loss** | BCE + Dice (combined, equal weight) |
| **Optimizer** | Adam (β1=0.9, β2=0.999) |
| **Threshold** | 0.5 |
| **Fine-tuning** | No — trained from scratch |
| **Val Dice (checkpoint)** | 0.8034 |
| **Val IoU (checkpoint)** | 0.6802 |
| **Config file** | `experiments/review_1/config.json` |
| **Results dir** | `results/review_1/` |

#### Test Results (Kvasir-SEG, 100 images)
| Metric | Mean | Std |
|---|---|---|
| **Dice** | **0.8120** | 0.2282 |
| **IoU** | **0.7307** | 0.2506 |
| **Precision** | **0.8532** | 0.2121 |
| **Recall** | 0.8520 | 0.2329 |
| **Specificity** | **0.9733** | 0.0485 |

**Purpose:** Establish the primary baseline for the Attention U-Net on Kvasir-SEG. Validate the full training pipeline end-to-end.

**Conclusion:** Strong in-domain performance. Dice 0.8120 exceeds the typical clinical threshold of 0.75. This checkpoint (`review_1_best.pth`) is the **project's canonical best model** and is used for all subsequent CVC external evaluations.

---

### EXP-02 — Attention U-Net + Dropout 0.2 (Kvasir-SEG, 50 Epochs)

| Field | Value |
|---|---|
| **Experiment type** | Training — regularisation ablation |
| **Architecture** | Attention U-Net |
| **Checkpoint** | `checkpoints/dropout_0.2/` |
| **Dataset (train)** | Kvasir-SEG |
| **Dataset (test)** | Kvasir-SEG test split (100 images) |
| **Dropout** | 0.2 (applied at bottleneck) |
| **Epochs** | 50 |
| **Batch size** | 8 |
| **Learning rate** | 1e-4 |
| **Weight decay** | 1e-5 |
| **Loss** | BCE + Dice |
| **Optimizer** | Adam |
| **Threshold** | 0.5 |
| **Fine-tuning** | No — trained from scratch |
| **Config file** | `experiments/dropout_0.2/config.json` |
| **Results dir** | `results/review_1_dropout/` |

#### Test Results (Kvasir-SEG, 100 images)
| Metric | Mean | Std |
|---|---|---|
| **Dice** | 0.8093 | 0.2325 |
| **IoU** | 0.7298 | 0.2603 |
| **Precision** | 0.8229 | 0.2401 |
| **Recall** | **0.8769** | 0.2127 |
| **Specificity** | 0.9650 | 0.0674 |

**Purpose:** Determine whether 0.2 dropout improves generalisation on Kvasir.

**Conclusion:** Dropout slightly reduces Dice (−0.0027) and Precision (−0.0303) but substantially improves Recall (+0.0249). Trade-off: more sensitive but less precise. Preferred if missing lesions is the greater risk. **Does not outperform EXP-01 on Dice/IoU.**

---

### EXP-03 — Standard U-Net + Dropout 0.2 (Kvasir-SEG, 30 Epochs)

| Field | Value |
|---|---|
| **Experiment type** | Training — architecture comparison |
| **Architecture** | Standard U-Net (no attention gates) |
| **Checkpoint** | `checkpoints/unet_dropout_30/` |
| **Dataset (train)** | Kvasir-SEG |
| **Dataset (test)** | Kvasir-SEG test split (100 images) |
| **Dropout** | 0.2 |
| **Epochs** | 30 |
| **Batch size** | 8 |
| **Learning rate** | 1e-4 |
| **Weight decay** | 1e-5 |
| **Loss** | BCE + Dice |
| **Optimizer** | Adam |
| **Threshold** | 0.5 |
| **Fine-tuning** | No — trained from scratch |
| **Config file** | `experiments/unet_dropout_30/config.json` |
| **Results dir** | `results/unet_dropout_30/` |

#### Test Results (Kvasir-SEG, 100 images)
| Metric | Mean | Std |
|---|---|---|
| **Dice** | 0.7929 | 0.2068 |
| **IoU** | 0.6976 | 0.2409 |
| **Precision** | 0.8177 | 0.2414 |
| **Recall** | 0.8483 | 0.1993 |
| **Specificity** | 0.9651 | 0.0640 |

**Purpose:** Establish Standard U-Net baseline for architecture comparison. Lowest std (0.2068) across Kvasir experiments.

**Conclusion:** Standard U-Net underperforms Attention U-Net by −1.91% Dice. Attention gates provide measurable benefit on Kvasir-SEG, even with fewer epochs.

---

### EXP-04 — Standard U-Net + Dropout 0.2 (Kvasir-SEG, 50 Epochs)

| Field | Value |
|---|---|
| **Experiment type** | Training — epoch-count sensitivity |
| **Architecture** | Standard U-Net (no attention gates) |
| **Checkpoint** | `checkpoints/unet_dropout_50/` |
| **Dataset (train)** | Kvasir-SEG |
| **Dataset (test)** | Kvasir-SEG test split (100 images) |
| **Dropout** | 0.2 |
| **Epochs** | 50 |
| **Batch size** | 8 |
| **Learning rate** | 1e-4 |
| **Weight decay** | 1e-5 |
| **Loss** | BCE + Dice |
| **Optimizer** | Adam |
| **Threshold** | 0.5 |
| **Fine-tuning** | No — trained from scratch |
| **Config file** | `experiments/unet_dropout_50/config.json` |
| **Results dir** | `results/unet_dropout_50/` |

#### Test Results (Kvasir-SEG, 100 images)
| Metric | Mean | Std |
|---|---|---|
| **Dice** | 0.8001 | 0.2450 |
| **IoU** | 0.7210 | 0.2689 |
| **Precision** | 0.8471 | 0.2265 |
| **Recall** | 0.8413 | 0.2462 |
| **Specificity** | 0.9716 | 0.0518 |

**Purpose:** Check whether more epochs help the Standard U-Net close the gap with EXP-01.

**Conclusion:** 20 additional epochs improve Std U-Net Dice by +0.72% and IoU by +2.34% over EXP-03. Still −1.19% below EXP-01 Attention U-Net (30ep). Standard U-Net is more epoch-sensitive than Attention U-Net.

---

### EXP-05 — CVC-ClinicDB Zero-Shot External Test

| Field | Value |
|---|---|
| **Experiment type** | External evaluation — zero-shot cross-dataset |
| **Architecture** | Attention U-Net (unchanged from EXP-01) |
| **Checkpoint used** | `checkpoints/review_1/review_1_best.pth` |
| **Checkpoint epoch** | 16 (Kvasir-trained, val_dice=0.8034) |
| **Retraining / fine-tuning** | **None. Weights frozen. Inference only.** |
| **Test dataset** | CVC-ClinicDB — all 612 image-mask pairs |
| **Preprocessing** | Identical to Kvasir pipeline (resize 256×256, ImageNet norm) |
| **CLAHE** | No |
| **Threshold** | 0.5 |
| **Results dir** | `results/cvc_external_test/` |
| **Script** | `experiments/cvc_external_test/cvc_evaluate.py` |

#### Pipeline Audit (conducted before/after run)
All 8 audit checks passed with **no implementation bug identified**:

| Check | Result |
|---|---|
| Image-mask pairing (612 pairs, stem-matched) | ✅ Correct |
| Mask binarization (RGB masks, >127 threshold) | ✅ Correct |
| Resize + interpolation (BILINEAR/NEAREST) | ✅ Correct |
| ImageNet normalization | ✅ Correct |
| Sigmoid + threshold 0.5 | ✅ Correct |
| Checkpoint identity (epoch=16, val_dice=0.8034) | ✅ Verified |
| Metric formulas (Dice/IoU/Prec/Rec/Spec) | ✅ Correct |
| Dataset version (official 612-frame CVC-ClinicDB) | ✅ Correct |

#### Test Results (CVC-ClinicDB, 612 images)
| Metric | Mean | Std |
|---|---|---|
| **Dice** | 0.4516 | 0.3196 |
| **IoU** | 0.3495 | 0.2849 |
| **Precision** | 0.5656 | 0.3535 |
| **Recall** | 0.6124 | 0.3881 |
| **Specificity** | 0.9296 | 0.0877 |

**Purpose:** Measure zero-shot generalisation of the Kvasir-trained model to an unseen colonoscopy device and institution.

**Conclusion:** Dice drops from 0.8120 (Kvasir) to 0.4516 (CVC). High std (±0.3196) indicates bimodal behaviour — model achieves Dice 0.89–0.97 on visually similar CVC frames, near 0.0 on dissimilar frames. Specificity remains high (0.93), confirming background identification is preserved. Result is a legitimate domain gap, not a pipeline bug. Note: base papers reporting ~78% Dice on CVC train **on** CVC — not directly comparable to this zero-shot result.

---

### EXP-06 — CVC Threshold Sensitivity Diagnostic

| Field | Value |
|---|---|
| **Experiment type** | Diagnostic — threshold sweep |
| **Architecture** | Attention U-Net (unchanged from EXP-01) |
| **Checkpoint used** | `checkpoints/review_1/review_1_best.pth` |
| **Retraining / fine-tuning** | **None. Single inference pass, thresholds swept in memory.** |
| **Test dataset** | CVC-ClinicDB — all 612 images |
| **Thresholds tested** | 0.1, 0.2, 0.3, 0.4, 0.5 |
| **Results dir** | `results/cvc_threshold_analysis/` |
| **Script** | `experiments/cvc_external_test/threshold_analysis.py` |

#### Results by Threshold (mean over 612 images)

| Threshold | Dice | IoU | Precision | Recall | Specificity |
|---|---|---|---|---|---|
| 0.1 | 0.4086 | 0.2980 | 0.3715 | **0.7776** | 0.8310 |
| 0.2 | 0.4344 | 0.3252 | 0.4495 | 0.7251 | 0.8773 |
| 0.3 | 0.4431 | 0.3358 | 0.4907 | 0.6930 | 0.8968 |
| 0.4 | 0.4490 | 0.3444 | 0.5226 | 0.6536 | 0.9147 |
| **0.5** *(official)* | **0.4516** | **0.3495** | **0.5656** | 0.6124 | **0.9296** |

Additional observations:
- Global probability map mean across all 612 images: **0.1485**
- Images with Dice~0 at threshold 0.5: **118/612 (19.3%)**
- Images with Dice~0 at threshold 0.1: **50/612 (8.2%)**
- Maximum Dice gain across all thresholds: **+0.0000** (threshold 0.5 is already optimal for Dice/IoU)

**Purpose:** Determine whether the low CVC Dice is a confidence calibration issue (threshold too high) or a genuine domain gap.

**Conclusion:** Threshold 0.5 is already the best for Dice and IoU. Lowering the threshold improves Recall but degrades Precision and Dice. The domain gap is **structural, not calibration-related**. 50 images cannot be detected even at threshold 0.1, confirming genuine feature-level mismatch.

---

### EXP-07 — CVC CLAHE Preprocessing Ablation

| Field | Value |
|---|---|
| **Experiment type** | Preprocessing ablation — inference only |
| **Architecture** | Attention U-Net (unchanged from EXP-01) |
| **Checkpoint used** | `checkpoints/review_1/review_1_best.pth` |
| **Retraining / fine-tuning** | **None. Preprocessing change only. Weights unchanged.** |
| **Test dataset** | CVC-ClinicDB — all 612 images |
| **CLAHE settings** | LAB colour space, L-channel only, clipLimit=2.0, tileGrid=8×8 |
| **Preprocessing pipeline** | CLAHE → resize 256×256 (BILINEAR) → ImageNet normalize |
| **Mask preprocessing** | Unchanged (identical to baseline) |
| **Threshold** | 0.5 |
| **Results dir** | `results/cvc_clahe_ablation/` |
| **Script** | `experiments/cvc_external_test/clahe_ablation.py` |

#### Test Results (CVC-ClinicDB, 612 images)
| Metric | Baseline (EXP-05) | CLAHE (EXP-07) | Delta |
|---|---|---|---|
| **Dice** | 0.4516 ± 0.3196 | **0.5702 ± 0.3020** | **+0.1186** |
| **IoU** | 0.3495 ± 0.2849 | **0.4594 ± 0.2916** | **+0.1099** |
| **Precision** | 0.5656 ± 0.3535 | **0.6121 ± 0.3260** | **+0.0465** |
| **Recall** | 0.6124 ± 0.3881 | **0.7372 ± 0.3411** | **+0.1248** |
| **Specificity** | 0.9296 ± 0.0877 | **0.9393 ± 0.0810** | **+0.0097** |

**Purpose:** Test whether local contrast enhancement (CLAHE) can bridge part of the CVC domain gap without any model change.

**Conclusion:** CLAHE produces a **+11.86% Dice improvement** — significant on all five metrics simultaneously. CVC-ClinicDB colonoscope frames have lower local contrast than Kvasir-SEG. CLAHE on the LAB L-channel redistributes contrast into a range the Kvasir-trained model recognises. The std also decreases (0.3196 → 0.3020), indicating fewer completely missed images. This confirms the domain gap is **partially a contrast/visibility problem**, not purely a structural feature mismatch.

---

### EXP-08 — CLAHE-Trained Attention U-Net (Kvasir-SEG, 30 Epochs)

| Field | Value |
|---|---|
| **Experiment type** | Training — contrast augmentation ablation |
| **Architecture** | Attention U-Net |
| **Checkpoint** | `checkpoints/review_1_clahe/review_1_best.pth` |
| **Dataset (train)** | Kvasir-SEG |
| **Dataset (test)** | Kvasir-SEG & CVC-ClinicDB |
| **Dropout** | None (0.0) |
| **Epochs** | 30 |
| **Batch size** | 8 |
| **Learning rate** | 1e-4 |
| **Weight decay** | 1e-5 |
| **Loss** | BCE + Dice |
| **Optimizer** | Adam |
| **Preprocessing** | CLAHE (LAB L-channel) applied to images |
| **Threshold** | 0.5 |
| **Config file** | `experiments/clahe_training/config.json` |
| **Results dir** | `results/review_1_clahe/` |

#### Test Results (Kvasir-SEG, 100 images)
| Metric | Mean | Std |
|---|---|---|
| **Dice** | 0.7953 | 0.2363 |
| **IoU** | 0.7106 | 0.2611 |
| **Precision** | 0.8461 | 0.2222 |
| **Recall** | 0.8350 | 0.2441 |
| **Specificity** | 0.9675 | 0.0663 |

#### Test Results (CVC-ClinicDB, 612 images)
| Metric | Mean | Std |
|---|---|---|
| **Dice** | **0.6568** | 0.3297 |
| **IoU** | **0.5676** | 0.3248 |
| **Precision** | **0.7897** | 0.2715 |
| **Recall** | 0.7149 | 0.3635 |
| **Specificity** | **0.9787** | 0.0389 |

**Purpose:** Measure whether training-time contrast augmentation (CLAHE) closes the domain gap further compared to inference-only CLAHE.

**Conclusion:** Training with CLAHE significantly improved CVC zero-shot Dice (0.4516 -> 0.6568), substantially outperforming the inference-only CLAHE (0.5702). Kvasir in-domain performance dropped very slightly (0.8120 -> 0.7953), which is a common and acceptable trade-off for improved generalization. This is the **best cross-domain model**.

---

## Final Comparison Table — All Experiments

### Kvasir-SEG In-Domain Results

| Exp | Architecture | Dropout | Epochs | Dice | IoU | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|---|---|
| EXP-01 | Attn U-Net | None | 30 | **0.8120±0.2282** | **0.7307±0.2506** | **0.8532±0.2121** | 0.8520±0.2329 | **0.9733±0.0485** |
| EXP-02 | Attn U-Net | 0.2 | 50 | 0.8093±0.2325 | 0.7298±0.2603 | 0.8229±0.2401 | **0.8769±0.2127** | 0.9650±0.0674 |
| EXP-03 | Std U-Net | 0.2 | 30 | 0.7929±0.2068 | 0.6976±0.2409 | 0.8177±0.2414 | 0.8483±0.1993 | 0.9651±0.0640 |
| EXP-04 | Std U-Net | 0.2 | 50 | 0.8001±0.2450 | 0.7210±0.2689 | 0.8471±0.2265 | 0.8413±0.2462 | 0.9716±0.0518 |
| EXP-08 | Attn U-Net (CLAHE) | None | 30 | 0.7953±0.2363 | 0.7106±0.2611 | 0.8461±0.2222 | 0.8350±0.2441 | 0.9675±0.0663 |

### CVC-ClinicDB External Results (Zero-Shot, 612 Images)

| Exp | Preprocessing | Threshold | Dice | IoU | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|---|
| EXP-05 | Standard (baseline) | 0.5 | 0.4516±0.3196 | 0.3495±0.2849 | 0.5656±0.3535 | 0.6124±0.3881 | 0.9296±0.0877 |
| EXP-06 | Standard (baseline) | 0.1 | 0.4086±0.2736 | 0.2980±0.2442 | 0.3715±0.3023 | 0.7776±0.3427 | 0.8310±0.1371 |
| EXP-06 | Standard (baseline) | 0.2 | 0.4344±0.2909 | 0.3252±0.2612 | 0.4495±0.3340 | 0.7251±0.3687 | 0.8773±0.1174 |
| EXP-06 | Standard (baseline) | 0.3 | 0.4431±0.3006 | 0.3358±0.2698 | 0.4907±0.3434 | 0.6930±0.3779 | 0.8968±0.1075 |
| EXP-06 | Standard (baseline) | 0.4 | 0.4490±0.3108 | 0.3444±0.2782 | 0.5226±0.3502 | 0.6536±0.3847 | 0.9147±0.0974 |
| EXP-07 | CLAHE (inference only) | 0.5 | 0.5702±0.3020 | 0.4594±0.2916 | 0.6121±0.3260 | 0.7372±0.3411 | 0.9393±0.0810 |
| **EXP-08** | **CLAHE (train + inference)** | **0.5** | **0.6568±0.3297** | **0.5676±0.3248** | **0.7897±0.2715** | **0.7149±0.3635** | **0.9787±0.0389** |

---

## Current Best Configuration

### Best In-Domain (Kvasir-SEG)
**EXP-01 — Attention U-Net, 30 epochs, no dropout**

| Setting | Value |
|---|---|
| Architecture | Attention U-Net (7,851,773 params) |
| Checkpoint | `checkpoints/review_1/review_1_best.pth` |
| Epochs trained | 30 (best at epoch 16) |
| Dropout | None |
| Loss | BCE + Dice |
| Optimizer | Adam, LR=1e-4, WD=1e-5 |
| Batch size | 8 |
| Threshold | 0.5 |
| **Dice** | **0.8120 ± 0.2282** |
| **IoU** | **0.7307 ± 0.2506** |

### Best Cross-Domain (CVC-ClinicDB, Zero-Shot)
**EXP-08 — CLAHE-Trained Attention U-Net (evaluated with CLAHE)**

| Setting | Value |
|---|---|
| Architecture | Attention U-Net (7,851,773 params) |
| Checkpoint | `checkpoints/review_1_clahe/review_1_best.pth` |
| Preprocessing | CLAHE on LAB L-channel (clip=2.0, tile=8×8) during training and inference |
| Threshold | 0.5 |
| **Dice** | **0.6568 ± 0.3297** |
| **IoU** | **0.5676 ± 0.3248** |
| Dice improvement over baseline | **+0.2052 (+45.4% relative)** |

> [!NOTE]
> Training the model with CLAHE augmentation significantly improved its ability to generalize to the CVC-ClinicDB dataset, substantially outperforming the inference-only CLAHE approach.

---

## Results File Index

| Experiment | Metrics JSON | Config |
|---|---|---|
| EXP-01 | `results/review_1/metrics.json` | `experiments/review_1/config.json` |
| EXP-02 | `results/review_1_dropout/metrics.json` | `experiments/dropout_0.2/config.json` |
| EXP-03 | `results/unet_dropout_30/metrics.json` | `experiments/unet_dropout_30/config.json` |
| EXP-04 | `results/unet_dropout_50/metrics.json` | `experiments/unet_dropout_50/config.json` |
| EXP-05 | `results/cvc_external_test/metrics.json` | `experiments/cvc_external_test/cvc_evaluate.py` |
| EXP-06 | `results/cvc_threshold_analysis/threshold_comparison.json` | `experiments/cvc_external_test/threshold_analysis.py` |
| EXP-07 | `results/cvc_clahe_ablation/metrics.json` | `experiments/cvc_external_test/clahe_ablation.py` |
| EXP-08 | `results/review_1_clahe/metrics.json` | `experiments/clahe_training/run_exp_08.py` |
