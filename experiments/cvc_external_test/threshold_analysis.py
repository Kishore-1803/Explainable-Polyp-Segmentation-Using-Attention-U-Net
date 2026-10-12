"""
CVC-ClinicDB Threshold Sensitivity Analysis
============================================
Runs ONE forward pass through all 612 CVC images, stores probability maps,
then evaluates at thresholds 0.1, 0.2, 0.3, 0.4, 0.5 — no retraining,
no preprocessing changes, no model modification.

Saves results to: results/cvc_threshold_analysis/
  - threshold_comparison.json
  - threshold_table.txt
  - threshold_sensitivity_plot.png
  - threshold_dice_distribution.png

Does NOT modify results/cvc_external_test/ (official 0.5 baseline).
"""

import sys, json
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths (identical to cvc_evaluate.py) ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
CHECKPOINT   = PROJECT_ROOT / "checkpoints" / "review_1" / "review_1_best.pth"
CVC_IMG_DIR  = PROJECT_ROOT / "cvc_data" / "PNG" / "Original"
CVC_MSK_DIR  = PROJECT_ROOT / "cvc_data" / "PNG" / "Ground Truth"
OUT_DIR      = PROJECT_ROOT / "results" / "cvc_threshold_analysis"

sys.path.insert(0, str(SRC_DIR))

MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
IMAGE_SIZE = (256, 256)
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Metric computation ─────────────────────────────────────────────────────────
def compute_metrics(pred_prob: np.ndarray, gt: np.ndarray, thr: float) -> dict:
    pred = pred_prob > thr
    gt_b = gt > 0.5
    tp = np.sum( pred &  gt_b)
    fp = np.sum( pred & ~gt_b)
    fn = np.sum(~pred &  gt_b)
    tn = np.sum(~pred & ~gt_b)
    e  = 1e-7
    return {
        "dice":        float((2*tp + e) / (2*tp + fp + fn + e)),
        "iou":         float((tp + e)   / (tp + fp + fn + e)),
        "precision":   float((tp + e)   / (tp + fp + e)),
        "recall":      float((tp + e)   / (tp + fn + e)),
        "specificity": float((tn + e)   / (tn + fp + e)),
    }

# ── Load model ─────────────────────────────────────────────────────────────────
def load_model():
    from models.attention_unet import AttentionUNet
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = AttentionUNet(in_channels=3, out_channels=1, dropout_rate=0.0)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    meta = {k: v for k, v in state.items() if k not in ("model_state_dict", "optimizer_state_dict")}
    print(f"[model] Loaded checkpoint — epoch={meta.get('epoch','?')}  "
          f"val_dice={meta.get('val_dice', '?'):.4f}")
    return model

# ── Preprocessing ──────────────────────────────────────────────────────────────
def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, IMAGE_SIZE, interpolation=TF.InterpolationMode.BILINEAR)
    t   = TF.to_tensor(img)
    return TF.normalize(t, MEAN, STD)

def load_mask(path: Path) -> np.ndarray:
    msk = Image.open(path).convert("L")
    msk = TF.resize(msk, IMAGE_SIZE, interpolation=TF.InterpolationMode.NEAREST)
    return (np.array(msk, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

# ── Summary stats ──────────────────────────────────────────────────────────────
def summarise(per_image: list[dict]) -> dict:
    keys = ["dice", "iou", "precision", "recall", "specificity"]
    out  = {}
    for k in keys:
        vals = [m[k] for m in per_image]
        out[f"mean_{k}"] = float(np.mean(vals))
        out[f"std_{k}"]  = float(np.std(vals))
        out[f"median_{k}"] = float(np.median(vals))
    return out

# ── Printing ───────────────────────────────────────────────────────────────────
def print_table(results: dict):
    keys   = ["dice", "iou", "precision", "recall", "specificity"]
    labels = ["Dice", "IoU", "Precision", "Recall", "Specificity"]
    header = f"{'Threshold':<12}" + "".join(f"{l:<22}" for l in labels)
    print("\n" + "="*len(header))
    print("  CVC-ClinicDB — Threshold Sensitivity (mean +/- std)")
    print("="*len(header))
    print(header)
    print("-"*len(header))
    for thr, s in sorted(results.items()):
        row = f"{thr:<12}"
        for k in keys:
            row += f"{s[f'mean_{k}']:.4f} +/- {s[f'std_{k}']:.4f}   "
        print(row)
    print("="*len(header) + "\n")

# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_line_chart(results: dict, save_path: Path):
    thrs   = sorted(results.keys())
    keys   = ["dice", "iou", "precision", "recall", "specificity"]
    labels = ["Dice", "IoU", "Precision", "Recall", "Specificity"]
    colors = ["#2196F3", "#4CAF50", "#F44336", "#9C27B0", "#FF9800"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, lbl, col in zip(keys, labels, colors):
        means = [results[t][f"mean_{k}"] for t in thrs]
        stds  = [results[t][f"std_{k}"]  for t in thrs]
        ax.plot(thrs, means, marker="o", label=lbl, color=col, linewidth=2)
        ax.fill_between(thrs,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.12, color=col)

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("CVC-ClinicDB: Threshold Sensitivity — Kvasir-trained Attention U-Net\n"
                 "(shaded band = mean ± 1 std)", fontsize=11)
    ax.set_xticks(thrs)
    ax.set_ylim(0, 1.05)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Official threshold (0.5)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved line chart -> {save_path}")


def plot_dice_distribution(all_probs: list, all_gts: list, save_path: Path):
    """
    Violin plot showing per-image Dice distribution at each threshold.
    Reveals whether the spread is due to calibration or structural domain gap.
    """
    thrs   = [0.1, 0.2, 0.3, 0.4, 0.5]
    data   = []
    for thr in thrs:
        dices = [compute_metrics(p, g, thr)["dice"] for p, g in zip(all_probs, all_gts)]
        data.append(dices)

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data, positions=thrs, widths=0.06, showmedians=True,
                          showextrema=True)
    colors = ["#42A5F5", "#26C6DA", "#66BB6A", "#FFA726", "#EF5350"]
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    # Add box overlay
    ax.boxplot(data, positions=thrs, widths=0.025, patch_artist=False,
               medianprops=dict(color="black", linewidth=2),
               whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
               flierprops=dict(marker=".", markersize=2, alpha=0.4))

    # Annotate medians
    for thr, d in zip(thrs, data):
        med = float(np.median(d))
        ax.text(thr, med + 0.035, f"{med:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.axhline(y=0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Per-Image Dice Score", fontsize=12)
    ax.set_title("Per-Image Dice Distribution at Each Threshold\n"
                 "(violin = distribution shape; box = IQR; line = median)", fontsize=11)
    ax.set_xticks(thrs)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved distribution violin -> {save_path}")


def plot_zero_dice_analysis(all_probs: list, all_gts: list, save_path: Path):
    """
    Bar chart: how many images have Dice=0 at each threshold (model gives 0 predictions).
    Key diagnostic for calibration vs domain gap.
    """
    thrs  = [0.1, 0.2, 0.3, 0.4, 0.5]
    zeros = []
    low   = []  # Dice < 0.5
    high  = []  # Dice >= 0.5

    for thr in thrs:
        dices = [compute_metrics(p, g, thr)["dice"] for p, g in zip(all_probs, all_gts)]
        zeros.append(sum(1 for d in dices if d < 0.01))
        low.append(sum(1 for d in dices if 0.01 <= d < 0.5))
        high.append(sum(1 for d in dices if d >= 0.5))

    x = np.arange(len(thrs))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w,   zeros, w, label="Dice~0 (no prediction)", color="#EF5350", edgecolor="black", alpha=0.85)
    ax.bar(x,       low,   w, label="0 < Dice < 0.5 (partial)", color="#FFA726", edgecolor="black", alpha=0.85)
    ax.bar(x + w,   high,  w, label="Dice >= 0.5 (good)", color="#66BB6A", edgecolor="black", alpha=0.85)

    for bars in [
        ax.containers[0],
        ax.containers[1],
        ax.containers[2],
    ]:
        ax.bar_label(bars, fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"thr={t}" for t in thrs], fontsize=10)
    ax.set_ylabel("Number of CVC Images (out of 612)", fontsize=11)
    ax.set_title("CVC Images by Dice Quality at Each Threshold\n"
                 "(diagnostic: Dice~0 means model never reaches threshold confidence)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved zero-dice bar chart -> {save_path}")


def save_text_table(results: dict, save_path: Path):
    keys   = ["dice", "iou", "precision", "recall", "specificity"]
    labels = ["Dice", "IoU", "Precision", "Recall", "Specificity"]
    lines  = [
        "CVC-ClinicDB Threshold Sensitivity Analysis",
        "Model: Attention U-Net (Kvasir-trained, 30ep, no dropout)",
        "Checkpoint: review_1_best.pth",
        "Images: 612 (all CVC-ClinicDB)",
        "="*90,
        f"{'Threshold':<12}" + "".join(f"{l+' (mean+/-std)':<25}" for l in labels),
        "-"*90,
    ]
    for thr in sorted(results.keys()):
        s   = results[thr]
        row = f"{thr:<12}"
        for k in keys:
            cell = f"{s[f'mean_{k}']:.4f} +/- {s[f'std_{k}']:.4f}"
            row += f"{cell:<25}"
        lines.append(row)
    lines.append("="*90)
    lines.append("")
    lines.append("Best Dice threshold: " +
                 str(max(results, key=lambda t: results[t]["mean_dice"])))
    lines.append("Best IoU  threshold: " +
                 str(max(results, key=lambda t: results[t]["mean_iou"])))
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out]  Saved text table -> {save_path}")


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  CVC-ClinicDB Threshold Sensitivity Analysis")
    print(f"  Device: {DEVICE}  |  Thresholds: {THRESHOLDS}")
    print("="*65 + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify dataset present
    imgs = sorted(CVC_IMG_DIR.glob("*.png"), key=lambda x: int(x.stem))
    msks = sorted(CVC_MSK_DIR.glob("*.png"), key=lambda x: int(x.stem))
    assert len(imgs) == 612 and len(msks) == 612, \
        f"Expected 612 pairs, got {len(imgs)} images / {len(msks)} masks"
    print(f"[data] {len(imgs)} CVC image-mask pairs ready.")

    # Load model once
    model = load_model()

    # ── Single inference pass — collect all probability maps ──────────────────
    print(f"\n[infer] Running inference on all {len(imgs)} images (single pass)...\n")
    all_probs = []
    all_gts   = []

    with torch.no_grad():
        for img_path, msk_path in tqdm(zip(imgs, msks), total=len(imgs), desc="Inference"):
            img_t = load_image(img_path).unsqueeze(0).to(DEVICE)
            logit = model(img_t)
            prob  = torch.sigmoid(logit)[0, 0].cpu().numpy()   # (256,256), float32
            gt    = load_mask(msk_path)                         # (256,256), binary float32
            all_probs.append(prob)
            all_gts.append(gt)

    all_probs = np.array(all_probs, dtype=np.float32)  # (612, 256, 256)
    all_gts   = np.array(all_gts,   dtype=np.float32)  # (612, 256, 256)
    print(f"\n[infer] Done. Probability map stats: "
          f"min={all_probs.min():.4f}  max={all_probs.max():.4f}  "
          f"mean={all_probs.mean():.4f}")

    # ── Threshold sweep ────────────────────────────────────────────────────────
    print("\n[eval] Sweeping thresholds...\n")
    results    = {}   # thr -> summary dict
    per_image  = {}   # thr -> list of per-image dicts

    for thr in THRESHOLDS:
        img_metrics = [
            compute_metrics(all_probs[i], all_gts[i], thr)
            for i in range(len(imgs))
        ]
        summary = summarise(img_metrics)
        results[thr]   = summary
        per_image[thr] = img_metrics

    # ── Print table ────────────────────────────────────────────────────────────
    print_table(results)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_out = {
        "experiment": "CVC-ClinicDB Threshold Sensitivity Analysis",
        "model": "Attention U-Net (Kvasir-trained, 30ep, no dropout)",
        "checkpoint": str(CHECKPOINT),
        "dataset": "CVC-ClinicDB — all 612 images",
        "image_size": list(IMAGE_SIZE),
        "normalization": {"mean": MEAN, "std": STD},
        "thresholds_evaluated": THRESHOLDS,
        "results": {str(t): results[t] for t in THRESHOLDS},
        "per_image": {
            str(t): [
                {"image": imgs[i].name, **per_image[t][i]}
                for i in range(len(imgs))
            ]
            for t in THRESHOLDS
        },
    }
    json_path = OUT_DIR / "threshold_comparison.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=4)
    print(f"[out]  Saved threshold_comparison.json -> {json_path}")

    # ── Save text table ────────────────────────────────────────────────────────
    save_text_table(results, OUT_DIR / "threshold_table.txt")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_line_chart(results, OUT_DIR / "threshold_sensitivity_plot.png")
    plot_dice_distribution(all_probs, all_gts, OUT_DIR / "threshold_dice_distribution.png")
    plot_zero_dice_analysis(all_probs, all_gts, OUT_DIR / "threshold_zero_dice_analysis.png")

    # ── Final diagnostic conclusion ────────────────────────────────────────────
    best_thr_dice = max(results, key=lambda t: results[t]["mean_dice"])
    best_dice     = results[best_thr_dice]["mean_dice"]
    dice_at_05    = results[0.5]["mean_dice"]
    gain          = best_dice - dice_at_05

    zeros_at_05  = sum(1 for m in per_image[0.5] if m["dice"] < 0.01)
    zeros_at_01  = sum(1 for m in per_image[0.1] if m["dice"] < 0.01)

    print("="*65)
    print("  DIAGNOSTIC CONCLUSION")
    print("="*65)
    print(f"  Dice at threshold=0.5  (official baseline): {dice_at_05:.4f}")
    print(f"  Best Dice at threshold={best_thr_dice}:                {best_dice:.4f}")
    print(f"  Gain from optimal threshold:                +{gain:.4f}")
    print()
    print(f"  Images with Dice~0 at threshold=0.5: {zeros_at_05}/612  ({100*zeros_at_05/612:.1f}%)")
    print(f"  Images with Dice~0 at threshold=0.1: {zeros_at_01}/612  ({100*zeros_at_01/612:.1f}%)")
    print()
    if gain > 0.10:
        print("  VERDICT: Significant calibration issue — lowering threshold")
        print("           substantially improves Dice, suggesting model outputs")
        print("           low-confidence correct predictions on CVC.")
    elif gain > 0.03:
        print("  VERDICT: Moderate calibration effect — some improvement from")
        print("           lower threshold, but domain gap is the primary factor.")
    else:
        print("  VERDICT: Threshold has minimal effect — performance is limited")
        print("           by genuine domain gap, not calibration.")
    print("="*65 + "\n")
    print(f"[done] All results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
