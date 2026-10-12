"""
CVC-ClinicDB CLAHE Preprocessing Ablation
==========================================
Compares two preprocessing pipelines on the same 612 CVC images:
  - Baseline : resize → normalize  (existing official result)
  - CLAHE    : CLAHE (LAB L-channel) → resize → normalize

Model, weights, architecture, threshold, mask handling — ALL UNCHANGED.
No retraining, no fine-tuning, no modification of existing results.

Outputs (all in results/cvc_clahe_ablation/):
  predictions/          - 612 predicted binary masks
  visualizations/       - 20 sample 4-panel grids
  clahe_previews/       - 10 before/after CLAHE inspection images
  metrics.json          - full per-image + summary for CLAHE variant
  baseline_vs_clahe_comparison.json
  baseline_vs_clahe_table.txt
  baseline_vs_clahe_chart.png
  clahe_summary_chart.png
"""

import sys, json
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
CHECKPOINT   = PROJECT_ROOT / "checkpoints" / "review_1" / "review_1_best.pth"
CVC_IMG_DIR  = PROJECT_ROOT / "cvc_data" / "PNG" / "Original"
CVC_MSK_DIR  = PROJECT_ROOT / "cvc_data" / "PNG" / "Ground Truth"
OUT_DIR      = PROJECT_ROOT / "results" / "cvc_clahe_ablation"
PRED_DIR     = OUT_DIR / "predictions"
VIZ_DIR      = OUT_DIR / "visualizations"
PREV_DIR     = OUT_DIR / "clahe_previews"

sys.path.insert(0, str(SRC_DIR))

# ── Constants (identical to official baseline) ──────────────────────────────
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
IMAGE_SIZE = (256, 256)
THRESHOLD  = 0.5
NUM_VIZ    = 20
NUM_PREV   = 10          # before/after CLAHE inspection images
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLAHE parameters
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)

# Baseline results (from official cvc_external_test run at threshold=0.5)
BASELINE = {
    "mean_dice":        0.4516, "std_dice":        0.3196,
    "mean_iou":         0.3495, "std_iou":         0.2849,
    "mean_precision":   0.5656, "std_precision":   0.3535,
    "mean_recall":      0.6124, "std_recall":      0.3881,
    "mean_specificity": 0.9296, "std_specificity": 0.0877,
}

# ── CLAHE preprocessing ───────────────────────────────────────────────────────
def apply_clahe_rgb(pil_image: Image.Image) -> Image.Image:
    """
    Apply CLAHE to the L channel of the LAB colour space, return RGB PIL image.
    This enhances local contrast while preserving colour balance — standard
    approach for medical image contrast enhancement.
    """
    img_rgb  = np.array(pil_image, dtype=np.uint8)          # H x W x 3, uint8
    img_lab  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)     # -> LAB
    l, a, b  = cv2.split(img_lab)

    clahe    = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                tileGridSize=CLAHE_TILE_GRID)
    l_eq     = clahe.apply(l)                                # enhance L only

    lab_eq   = cv2.merge([l_eq, a, b])
    img_eq   = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)      # back to RGB
    return Image.fromarray(img_eq)

# ── Image loading pipelines ──────────────────────────────────────────────────
def load_image_clahe(path: Path) -> torch.Tensor:
    """CLAHE → resize → to_tensor → normalize"""
    img   = Image.open(path).convert("RGB")
    img   = apply_clahe_rgb(img)                             # <-- CLAHE step
    img   = TF.resize(img, IMAGE_SIZE, interpolation=TF.InterpolationMode.BILINEAR)
    t     = TF.to_tensor(img)
    return TF.normalize(t, MEAN, STD)

def load_mask(path: Path) -> np.ndarray:
    msk = Image.open(path).convert("L")
    msk = TF.resize(msk, IMAGE_SIZE, interpolation=TF.InterpolationMode.NEAREST)
    return (np.array(msk, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(prob: np.ndarray, gt: np.ndarray, thr: float = THRESHOLD) -> dict:
    pred = prob > thr
    gt_b = gt  > 0.5
    tp   = np.sum( pred &  gt_b)
    fp   = np.sum( pred & ~gt_b)
    fn   = np.sum(~pred &  gt_b)
    tn   = np.sum(~pred & ~gt_b)
    e    = 1e-7
    return {
        "dice":        float((2*tp+e) / (2*tp+fp+fn+e)),
        "iou":         float((tp+e)   / (tp+fp+fn+e)),
        "precision":   float((tp+e)   / (tp+fp+e)),
        "recall":      float((tp+e)   / (tp+fn+e)),
        "specificity": float((tn+e)   / (tn+fp+e)),
    }

def summarise(per_image: list) -> dict:
    keys = ["dice", "iou", "precision", "recall", "specificity"]
    out  = {}
    for k in keys:
        v = [m[k] for m in per_image]
        out[f"mean_{k}"] = float(np.mean(v))
        out[f"std_{k}"]  = float(np.std(v))
    return out

# ── Unnormalise for display ───────────────────────────────────────────────────
def unnorm(t: torch.Tensor) -> np.ndarray:
    img = t.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)

# ── Save prediction mask ──────────────────────────────────────────────────────
def save_pred_mask(prob: np.ndarray, path: Path):
    Image.fromarray(((prob > THRESHOLD).astype(np.uint8) * 255), mode="L").save(path)

# ── 4-panel visualisation ─────────────────────────────────────────────────────
def save_viz(img_t, gt, prob, m, path, name):
    img_rgb   = unnorm(img_t)
    img_u8    = (img_rgb * 255).astype(np.uint8)
    pred_bin  = (prob > THRESHOLD).astype(np.uint8)
    overlay   = img_u8.copy()
    contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"{name}  |  Dice={m['dice']:.4f}  IoU={m['iou']:.4f}  "
        f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  Spec={m['specificity']:.4f}",
        fontsize=8, y=1.01
    )
    axes[0].imshow(img_rgb);              axes[0].set_title("CLAHE Image");    axes[0].axis("off")
    axes[1].imshow(gt, cmap="gray");      axes[1].set_title("Ground Truth");   axes[1].axis("off")
    axes[2].imshow(pred_bin, cmap="gray"); axes[2].set_title("Prediction");    axes[2].axis("off")
    axes[3].imshow(overlay[:, :, ::-1]);  axes[3].set_title("Boundary Overlay"); axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()

# ── CLAHE inspection: before / after side-by-side ────────────────────────────
def save_clahe_preview(img_path: Path, save_path: Path):
    """
    3-panel: Original | L-channel before CLAHE | L-channel after CLAHE
    + 4th panel: CLAHE RGB for visual comparison.
    """
    orig      = Image.open(img_path).convert("RGB")
    orig_np   = np.array(orig, dtype=np.uint8)

    lab_before = cv2.cvtColor(orig_np, cv2.COLOR_RGB2LAB)
    l_before, a, b = cv2.split(lab_before)

    clahe     = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_after   = clahe.apply(l_before)

    lab_after = cv2.merge([l_after, a, b])
    eq_np     = cv2.cvtColor(lab_after, cv2.COLOR_LAB2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"CLAHE Inspection — {img_path.name}", fontsize=11)
    axes[0].imshow(orig_np);   axes[0].set_title("Original RGB");       axes[0].axis("off")
    axes[1].imshow(l_before, cmap="gray"); axes[1].set_title("L-channel (before CLAHE)"); axes[1].axis("off")
    axes[2].imshow(l_after,  cmap="gray"); axes[2].set_title("L-channel (after CLAHE)");  axes[2].axis("off")
    axes[3].imshow(eq_np);     axes[3].set_title("CLAHE RGB result");    axes[3].axis("off")

    # Pixel stats
    axes[1].set_xlabel(f"mean={l_before.mean():.1f}  std={l_before.std():.1f}", fontsize=8)
    axes[2].set_xlabel(f"mean={l_after.mean():.1f}   std={l_after.std():.1f}", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()

# ── Comparison chart ──────────────────────────────────────────────────────────
def plot_comparison(baseline: dict, clahe_s: dict, save_path: Path):
    mkeys  = ["dice", "iou", "precision", "recall", "specificity"]
    labels = ["Dice", "IoU", "Precision", "Recall", "Specificity"]
    x = np.arange(len(labels))
    w = 0.35

    b_means = [baseline[f"mean_{k}"] for k in mkeys]
    b_stds  = [baseline[f"std_{k}"]  for k in mkeys]
    c_means = [clahe_s[f"mean_{k}"]  for k in mkeys]
    c_stds  = [clahe_s[f"std_{k}"]   for k in mkeys]

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w/2, b_means, w, yerr=b_stds, label="Baseline (no CLAHE)",
                capsize=5, color="#4C72B0", alpha=0.85, edgecolor="black")
    b2 = ax.bar(x + w/2, c_means, w, yerr=c_stds, label="CLAHE (LAB L-channel)",
                capsize=5, color="#55A868", alpha=0.85, edgecolor="black")

    for bar, val in zip(b1, b_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#4C72B0")
    for bar, val in zip(b2, c_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#2a7a40")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("CVC-ClinicDB: Baseline vs CLAHE Preprocessing\n"
                 "Attention U-Net (Kvasir-trained, 30ep) — Zero-Shot Transfer",
                 fontsize=11, fontweight="bold")
    ax.axhline(y=0.75, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label="Clinical threshold (0.75)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved comparison chart -> {save_path}")

def plot_clahe_summary(clahe_s: dict, save_path: Path):
    mkeys  = ["dice", "iou", "precision", "recall", "specificity"]
    labels = ["Dice", "IoU", "Precision", "Recall", "Specificity"]
    means  = [clahe_s[f"mean_{k}"] for k in mkeys]
    stds   = [clahe_s[f"std_{k}"]  for k in mkeys]
    colors = ["#2196F3", "#4CAF50", "#F44336", "#9C27B0", "#FF9800"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors,
                  alpha=0.85, edgecolor="black")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.008,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("CVC-ClinicDB CLAHE Variant — Attention U-Net (Kvasir-trained, 30ep)",
                 fontsize=12)
    ax.axhline(y=0.75, color="red", linestyle="--", linewidth=1,
               label="Clinical threshold (0.75)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved CLAHE summary chart -> {save_path}")

def save_text_table(baseline: dict, clahe_s: dict, save_path: Path):
    mkeys  = ["dice", "iou", "precision", "recall", "specificity"]
    labels = ["Dice", "IoU", "Precision", "Recall", "Specificity"]
    lines  = [
        "CVC-ClinicDB: Baseline vs CLAHE Preprocessing Ablation",
        "Model: Attention U-Net (Kvasir-trained, 30ep, no dropout)",
        "Checkpoint: review_1_best.pth | Threshold: 0.5 | Images: 612",
        "CLAHE: LAB L-channel, clipLimit=2.0, tileGrid=8x8",
        "="*80,
        f"{'Metric':<16} {'Baseline (mean+/-std)':<28} {'CLAHE (mean+/-std)':<28} {'Delta':<10}",
        "-"*80,
    ]
    for k, lbl in zip(mkeys, labels):
        bm, bs = baseline[f"mean_{k}"], baseline[f"std_{k}"]
        cm, cs = clahe_s[f"mean_{k}"],  clahe_s[f"std_{k}"]
        delta  = cm - bm
        sign   = "+" if delta >= 0 else ""
        lines.append(
            f"{lbl:<16} {bm:.4f} +/- {bs:.4f}         {cm:.4f} +/- {cs:.4f}         {sign}{delta:.4f}"
        )
    lines.append("="*80)
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out]  Saved text table -> {save_path}")

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  CVC-ClinicDB CLAHE Preprocessing Ablation")
    print(f"  Device: {DEVICE}  |  CLAHE clip={CLAHE_CLIP_LIMIT}  tile={CLAHE_TILE_GRID}")
    print("="*65 + "\n")

    for d in [OUT_DIR, PRED_DIR, VIZ_DIR, PREV_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load image/mask paths
    imgs = sorted(CVC_IMG_DIR.glob("*.png"), key=lambda x: int(x.stem))
    msks = sorted(CVC_MSK_DIR.glob("*.png"), key=lambda x: int(x.stem))
    assert len(imgs) == 612 and len(msks) == 612
    print(f"[data] {len(imgs)} CVC image-mask pairs.\n")

    # ── CLAHE inspection FIRST (verify preprocessing is applied) ────────────
    print(f"[prev] Saving {NUM_PREV} CLAHE before/after inspection images...")
    prev_indices = np.linspace(0, 611, NUM_PREV, dtype=int).tolist()
    for idx in prev_indices:
        save_clahe_preview(imgs[idx], PREV_DIR / f"clahe_preview_{imgs[idx].stem}.png")
    print(f"[prev] Saved to {PREV_DIR}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    from models.attention_unet import AttentionUNet
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = AttentionUNet(in_channels=3, out_channels=1, dropout_rate=0.0)
    sd    = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    meta  = {k: v for k, v in state.items()
             if k not in ("model_state_dict", "optimizer_state_dict")}
    print(f"[model] Loaded — epoch={meta.get('epoch','?')}  "
          f"val_dice={meta.get('val_dice',0):.4f}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # ── Inference loop (CLAHE pipeline) ──────────────────────────────────────
    all_metrics   = []
    per_image_out = []
    viz_idx_set   = set(np.linspace(0, 611, NUM_VIZ, dtype=int).tolist())

    print(f"\n[eval] Running CLAHE inference on all {len(imgs)} CVC pairs...\n")

    with torch.no_grad():
        for idx, (img_path, msk_path) in enumerate(
                tqdm(zip(imgs, msks), total=len(imgs), desc="CLAHE Inference")):

            img_t = load_image_clahe(img_path)          # CLAHE → resize → norm
            gt    = load_mask(msk_path)                 # mask unchanged

            prob  = torch.sigmoid(
                model(img_t.unsqueeze(0).to(DEVICE))
            )[0, 0].cpu().numpy()

            m = compute_metrics(prob, gt)
            all_metrics.append(m)
            per_image_out.append({"image": img_path.name, "mask": msk_path.name, **m})

            # Save predicted mask
            save_pred_mask(prob, PRED_DIR / f"pred_{img_path.stem}.png")

            # Save visualisation
            if idx in viz_idx_set:
                save_viz(img_t, gt, prob, m,
                         VIZ_DIR / f"viz_{idx:04d}_{img_path.stem}.png",
                         img_path.stem)

    # ── Summarise ─────────────────────────────────────────────────────────────
    clahe_summary = summarise(all_metrics)

    # ── Print results ─────────────────────────────────────────────────────────
    mkeys  = ["dice", "iou", "precision", "recall", "specificity"]
    labels = ["Dice", "IoU", "Precision", "Recall", "Specificity"]

    print("\n" + "="*65)
    print("  CLAHE Ablation — RESULTS vs BASELINE")
    print("="*65)
    print(f"  {'Metric':<16} {'Baseline':<22} {'CLAHE':<22} {'Delta'}")
    print("-"*65)
    for k, lbl in zip(mkeys, labels):
        bm, bs = BASELINE[f"mean_{k}"], BASELINE[f"std_{k}"]
        cm, cs = clahe_summary[f"mean_{k}"], clahe_summary[f"std_{k}"]
        delta  = cm - bm
        sign   = "+" if delta >= 0 else ""
        flag   = " <-- improved" if delta > 0.005 else (" <-- worse" if delta < -0.005 else "")
        print(f"  {lbl:<16} {bm:.4f}+/-{bs:.4f}      {cm:.4f}+/-{cs:.4f}      {sign}{delta:.4f}{flag}")
    print("="*65 + "\n")

    # ── Save metrics.json ─────────────────────────────────────────────────────
    metrics_dict = {
        "experiment": "CVC-ClinicDB CLAHE Preprocessing Ablation",
        "model": "Attention U-Net (Kvasir-trained, 30ep, no dropout)",
        "checkpoint": str(CHECKPOINT),
        "preprocessing": {
            "clahe_clip_limit": CLAHE_CLIP_LIMIT,
            "clahe_tile_grid": list(CLAHE_TILE_GRID),
            "color_space": "LAB (L-channel only)",
            "resize": list(IMAGE_SIZE),
            "normalization": {"mean": MEAN, "std": STD},
        },
        "threshold": THRESHOLD,
        "num_images": len(imgs),
        "summary": clahe_summary,
        "per_image": per_image_out,
    }
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[out]  Saved metrics.json -> {metrics_path}")

    # ── Save comparison JSON ──────────────────────────────────────────────────
    comparison = {
        "Baseline (no CLAHE, official)": BASELINE,
        "CLAHE (LAB L-channel, clip=2.0, tile=8x8)": clahe_summary,
    }
    comp_path = OUT_DIR / "baseline_vs_clahe_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=4)
    print(f"[out]  Saved comparison JSON -> {comp_path}")

    # ── Save text table ───────────────────────────────────────────────────────
    save_text_table(BASELINE, clahe_summary, OUT_DIR / "baseline_vs_clahe_table.txt")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_comparison(BASELINE, clahe_summary, OUT_DIR / "baseline_vs_clahe_chart.png")
    plot_clahe_summary(clahe_summary, OUT_DIR / "clahe_summary_chart.png")

    # ── Diagnostic conclusion ─────────────────────────────────────────────────
    dice_delta = clahe_summary["mean_dice"] - BASELINE["mean_dice"]
    print("="*65)
    print("  DIAGNOSTIC CONCLUSION")
    print("="*65)
    print(f"  Dice delta (CLAHE vs baseline): {'+' if dice_delta>=0 else ''}{dice_delta:.4f}")
    if dice_delta > 0.03:
        print("  >> SIGNIFICANT IMPROVEMENT: CLAHE substantially helps.")
        print("     CVC domain gap is partly a contrast/visibility issue.")
    elif dice_delta > 0.005:
        print("  >> MARGINAL IMPROVEMENT: CLAHE provides a small benefit.")
        print("     Domain gap is real but visibility is a minor factor.")
    elif dice_delta < -0.005:
        print("  >> CLAHE HURTS: Enhanced contrast does not help the model.")
        print("     The domain gap is structural, not contrast-related.")
    else:
        print("  >> NO MEANINGFUL CHANGE: CLAHE is neutral.")
        print("     Domain gap is unrelated to local contrast distribution.")
    print("="*65)
    print(f"\n[done] All outputs -> {OUT_DIR}")
    print(f"       Predictions   : {PRED_DIR}  (612 masks)")
    print(f"       Visualisations: {VIZ_DIR}   ({NUM_VIZ} panels)")
    print(f"       CLAHE previews: {PREV_DIR}   ({NUM_PREV} inspection images)")


if __name__ == "__main__":
    main()
