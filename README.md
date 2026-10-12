# Explainable Polyp Segmentation

**A replication study of Attention-Guided U-Net with Grad-CAM, and a diagnosis-driven improvement.**

We re-implement an IEEE Access 2025 paper at its reported capacity, reproduce its
in-domain result, fail to reproduce its cross-dataset result, diagnose why, and fix
part of the gap with two targeted modifications.

---

## Team

Department of Artificial Intelligence, School of AI, Amrita Vishwa Vidyapeetham
Final Major Project (MIP), Academic Year 2025–2026

| Name | Roll Number |
|---|---|
| Kishore B | CB.SC.U4AIE23139 |
| Akhilesh Mohanasundaram | CB.SC.U4AIE23111 |
| Charan Teja G | CB.SC.U4AIE23172 |
| Sai Charan M | CB.SC.U4AIE23143 |

---

## Base Paper

> D. Gupta and S. Benila, "Attention-Guided U-Net With Grad-CAM for Explainable Polyp
> Segmentation in Colonoscopy Images," *IEEE Access*, vol. 13, 2025.
> DOI: [10.1109/ACCESS.2025.3626159](https://doi.org/10.1109/ACCESS.2025.3626159)

---

## Headline Results

All rows share the same 100-image Kvasir-SEG test split and all 612 CVC-ClinicDB frames.
CVC-ClinicDB is **never** used for training or model selection.

| Configuration | Kvasir DSC | Kvasir IoU | CVC DSC | CVC IoU | CVC Prec. | Area |
|---|---|---|---|---|---|---|
| Gupta & Benila (reported) <sup>†</sup> | 0.8012 | 0.7878 | *0.7823* | *0.7654* | — | — |
| Replication (31.4M params) | 0.8301 | 0.7429 | 0.4589 | 0.3388 | 0.3889 | 2.18× |
| + SE channel attention | 0.8445 | 0.7622 | 0.5530 | 0.4386 | 0.5267 | 1.80× |
| + Deep supervision | 0.8415 | 0.7544 | 0.4426 | 0.3237 | 0.3670 | 2.29× |
| + SE + deep supervision | **0.8560** | **0.7746** | 0.5557 | 0.4368 | 0.4833 | 1.87× |
| CLAHE preprocessing only | 0.8059 | 0.7190 | 0.5978 | 0.5035 | 0.7439 | 1.00× |
| **+ SE + deep sup. + CLAHE** | 0.8531 | 0.7739 | **0.7015** | **0.6164** | **0.8445** | **0.92×** |

<sup>†</sup> Validation means over 5-fold cross-validation; all other rows are held-out
test means. Rows use the base paper's preprocessing unless CLAHE is named.
"Area" is mean predicted lesion area ÷ mean ground-truth area (1.00× ideal).

### What we found

1. **The in-domain claim replicates — and is exceeded.** 0.8301 DSC against the
   reported 0.8012, measured on a held-out test split rather than validation folds.
2. **The cross-dataset claim does not replicate.** 0.4589 DSC against the reported
   0.7823, and matching the paper's capacity, preprocessing, augmentation and
   schedule does not close it.
3. **The failure is over-segmentation, not missed detection.** The replicated model
   predicts 2.18× too much area on CVC-ClinicDB (precision 0.3889 at recall 0.8013)
   while staying calibrated in-domain at 1.06×. It is not illumination-driven: the
   correlation between per-image Dice and frame brightness is only +0.057.
4. **Two targeted fixes recover most of the gap.** SE channel attention corrects the
   feature response, luminance-channel CLAHE corrects the input distribution, and
   they are complementary — neither alone exceeds 0.60 CVC DSC, together they reach
   0.7015 for **+0.14% parameters**.
5. **Deep supervision alone hurts** cross-dataset performance (0.4426 vs. 0.4589) and
   helps only in combination with SE. We report this negative interaction.
6. **The paper's dataset description cannot be reproduced.** It attributes 8,000
   images and three polyp subtypes to Kvasir-SEG. Public Kvasir-SEG has 1,000 images
   and no subtype labels; 8,000 is the size of Kvasir v2, a *classification* dataset
   with no segmentation masks. The paper does not say how masks or subtype labels
   were obtained for the extra images.

---

## Datasets

| Dataset | Images | Role |
|---|---|---|
| [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) | 1,000 + masks | Train / validation / test (800/100/100, seed 42) |
| [CVC-ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb) | 612 + masks | External zero-shot test only |

Neither is tracked in Git. Download them into `Kvasir-SEG/` and `cvc_data/`.

---

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## Reproducing the results

```bash
# 1. Cache both preprocessing pipelines for both datasets (~1 min)
python experiments/paper_replication/build_cache.py

# 2. Faithful replication of the base paper
python experiments/paper_replication/train_paper.py    --arm paper
python experiments/paper_replication/evaluate_paper.py --arm paper

# 3. Architecture variants (held-fixed paper preprocessing)
python experiments/paper_replication/train_variant.py    --variant se
python experiments/paper_replication/train_variant.py    --variant ds
python experiments/paper_replication/train_variant.py    --variant se_ds
python experiments/paper_replication/evaluate_variant.py --variant se_ds

# 4. Best model: architecture + luminance CLAHE
python experiments/paper_replication/train_variant.py    --variant se_ds --arm ours
python experiments/paper_replication/evaluate_variant.py --variant se_ds --arm ours

# 5. Diagnosis, figures and summary table
python experiments/paper_replication/diagnose_cvc.py       --arm paper
python experiments/paper_replication/regenerate_figures.py --variant se_ds
python experiments/paper_replication/benchmark_latency.py
python experiments/paper_replication/collect_results.py
```

One training run takes roughly 18 minutes on an RTX 4060 Laptop GPU (8 GB) at
50 epochs, batch size 16, with FP16 mixed precision.

## Web application

```bash
python app.py        # http://localhost:5001
```

Serves the best checkpoint (`checkpoints/paper_var_se_ds_ours/best.pth`) with live
inference, Grad-CAM overlays, blur/exposure warnings, per-patient session tracking
and an exportable HTML report.

---

## Project Structure

```text
Codes/
├── src/                                 # Library code
│   ├── data/dataset.py                  # Paired image/mask Dataset
│   ├── data/split.py                    # Seed-locked 80/10/10 split
│   ├── preprocessing/transform.py       # Resize, augment, CLAHE
│   ├── preprocessing/advanced_transform.py  # Blur / exposure quality gate
│   ├── models/attention_unet.py         # DoubleConv + AttentionGate blocks
│   ├── models/unet.py                   # Plain U-Net
│   ├── training/trainer.py              # BCE + Dice loss, training loop
│   ├── evaluation/evaluator.py          # Dice, IoU, precision, recall, specificity
│   ├── evaluation/visualize.py          # Result grids
│   ├── evaluation/confidence.py         # Heuristic confidence score
│   ├── evaluation/report_generator.py   # Patient HTML report
│   └── explainability/grad_cam.py       # Segmentation Grad-CAM
│
├── experiments/paper_replication/       # ← the replication study
│   ├── paper_preprocessing.py           # The two preprocessing pipelines
│   ├── build_cache.py                   # Precompute both, for both datasets
│   ├── paper_model.py                   # Full-width Attention U-Net (31.39M)
│   ├── variants.py                      # SE block, deep supervision, Tversky loss
│   ├── train_paper.py / train_variant.py
│   ├── evaluate_paper.py / evaluate_variant.py
│   ├── diagnose_cvc.py                  # Over-segmentation diagnosis
│   ├── regenerate_figures.py            # Qualitative, Grad-CAM, threshold sweep
│   ├── benchmark_latency.py
│   └── collect_results.py               # Builds the comparison table
│
├── app.py + templates/                  # Flask web application
├── docs/                                # Audit reports and write-ups
├── requirements.txt
│
├── Kvasir-SEG/  cvc_data/               # Datasets        (not in Git)
├── _cache/                              # Preprocessed arrays (not in Git)
├── checkpoints/                         # Trained weights (not in Git)
├── results/                             # Metrics and plots  (not in Git)
└── _archive/                            # Superseded material (not in Git)
```

---

## Model and Training Configuration

| Setting | Value |
|---|---|
| Architecture | Attention U-Net, encoder 64→512, bottleneck 1024 |
| Parameters | 31,389,165 (replication) / 31,434,604 (+ SE + deep supervision) |
| Input | 256 × 256 RGB, ImageNet normalisation |
| Loss | BCE + Dice |
| Optimizer | Adam, lr 1e-4 → 1e-6 cosine, weight decay 1e-5 |
| Batch size | 16, FP16 mixed precision |
| Epochs | 50, early stopping patience 10, seed 42 |
| Augmentation | ±30° rotation, h/v flip, 0.8–1.2 zoom, ±20% brightness |
| Threshold | τ = 0.5 |
| Grad-CAM layer | `up4`, the deepest decoder up-convolution |
| Latency | 11.98 ms / frame (83.5 FPS) on RTX 4060 Laptop GPU |

**Preprocessing pipelines.** The base paper's: min–max rescaling, channel-wise RGB
histogram equalisation, CLAHE, Gaussian σ = 1.0, 3×3 median. Ours: CLAHE on the CIE
LAB L\* channel only (clip 2.0, 8×8 tiles), leaving a\*, b\* untouched. Both are
applied identically to training and test data.

---

## Limitations

- A 0.0808 DSC gap to the reported figure remains, and the best model still
  over-segments by 0.92×–1.87× depending on configuration.
- Single runs with a fixed seed; no variance across seeds, so differences of ~0.01
  DSC are not claimed to be significant.
- Channel widths, attention-gate intermediate dimensions and CLAHE parameters were
  inferred — the paper does not specify them.
- Two datasets at 256×256 only; no temporal/video context.
- The confidence score in the web app is a heuristic, not calibrated certainty.
