"""
CVC-ClinicDB External Generalization Test
==========================================
Tests the Kvasir-trained Attention U-Net (review_1_best.pth) on CVC-ClinicDB
WITHOUT any fine-tuning or weight modification.

Pipeline:
  - Downloads CVC-ClinicDB automatically via Kaggle API
  - Applies identical preprocessing as Kvasir pipeline (resize 256x256, ImageNet normalization)
  - Runs inference with sigmoid + threshold=0.5
  - Reports Dice, IoU, Precision, Recall, Specificity (mean ± std)
  - Saves predictions, sample visualizations, and metrics.json

Usage:
    python cvc_evaluate.py
    
Requirements: pip install kaggle opendatasets
Make sure ~/.kaggle/kaggle.json exists with your API credentials.
"""

import os
import sys
import json
import glob
import shutil
import zipfile
import subprocess
import traceback
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from tqdm import tqdm

# ─────────────────────────────────────────────
#  Path configuration
# ─────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent   # d:\MIP PROJECT
SRC_DIR        = PROJECT_ROOT / "src"
CHECKPOINT     = PROJECT_ROOT / "checkpoints" / "review_1" / "review_1_best.pth"
OUTPUT_DIR     = PROJECT_ROOT / "results" / "cvc_external_test"
CVC_CACHE_DIR  = PROJECT_ROOT / "cvc_data"          # raw downloaded dataset lives here
VIZ_DIR        = OUTPUT_DIR / "visualizations"
PRED_DIR       = OUTPUT_DIR / "predictions"

DATASET_SLUG   = "balraj98/cvcclinicdb"
IMAGE_SIZE     = (256, 256)
THRESHOLD      = 0.5
MEAN           = [0.485, 0.456, 0.406]
STD            = [0.229, 0.224, 0.225]
NUM_VIZ        = 20          # how many sample visualizations to save
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add src to path so we can import the model
sys.path.insert(0, str(SRC_DIR))

# ─────────────────────────────────────────────
#  Step 1 — Install kaggle if not present
# ─────────────────────────────────────────────
def ensure_packages():
    pkgs = ["kaggle", "opendatasets"]
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[setup] Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# ─────────────────────────────────────────────
#  Step 2 — Download CVC-ClinicDB
# ─────────────────────────────────────────────
def download_cvc():
    """Download CVC-ClinicDB using kaggle CLI. Falls back to opendatasets."""
    CVC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_images = list(CVC_CACHE_DIR.rglob("*.png")) + list(CVC_CACHE_DIR.rglob("*.jpg"))
    if len(existing_images) > 100:
        print(f"[data] CVC-ClinicDB already cached at {CVC_CACHE_DIR} ({len(existing_images)} files found).")
        return

    kaggle_dir        = Path.home() / ".kaggle"
    kaggle_json       = kaggle_dir / "kaggle.json"
    kaggle_token_file = kaggle_dir / "access_token"   # new KGAT_ token format
    creds_ok = kaggle_json.exists() or kaggle_token_file.exists()
    if not creds_ok:
        print("\n" + "="*65)
        print("  ACTION REQUIRED - Kaggle API Credentials Missing")
        print("="*65)
        print("  1. Go to: https://www.kaggle.com/settings/api")
        print("  2. Click 'Create New API Token' -> downloads kaggle.json")
        print(f"  3. Move kaggle.json to: {kaggle_dir}")
        print("  4. Re-run this script.")
        print("="*65 + "\n")
        sys.exit(1)

    # Try kaggle CLI first
    print(f"[data] Downloading {DATASET_SLUG} via kaggle CLI...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", DATASET_SLUG,
             "-p", str(CVC_CACHE_DIR),
             "--unzip"],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("[data] Download complete via kaggle CLI.")
            return
        else:
            print(f"[data] kaggle CLI failed: {result.stderr.strip()}")
    except Exception as e:
        print(f"[data] kaggle CLI error: {e}")

    # Fallback: opendatasets
    print("[data] Trying opendatasets fallback...")
    try:
        import opendatasets as od
        od.download(f"https://www.kaggle.com/datasets/{DATASET_SLUG}", str(CVC_CACHE_DIR))
        print("[data] Download complete via opendatasets.")
    except Exception as e:
        print(f"[data] opendatasets also failed: {e}")
        sys.exit(1)

# ─────────────────────────────────────────────
#  Step 3 — Discover image/mask pairs in CVC
# ─────────────────────────────────────────────
def find_cvc_pairs():
    """
    CVC-ClinicDB (balraj98/cvcclinicdb) on Kaggle has this exact structure:
        cvc_data/
          PNG/
            Original/        <- 612 polyp images (.png)
            Ground Truth/    <- 612 binary masks  (.png)
          TIF/               <- duplicate set in TIF; we ignore this
    Returns sorted list of (image_path, mask_path) tuples.
    """
    # Use only the PNG versions — clean RGB images and binary masks
    img_dir = CVC_CACHE_DIR / "PNG" / "Original"
    msk_dir = CVC_CACHE_DIR / "PNG" / "Ground Truth"

    if not img_dir.exists() or not msk_dir.exists():
        # Fallback: scan recursively for PNG folders
        print("[data] Standard PNG folders not found, scanning recursively...")
        all_png = sorted(CVC_CACHE_DIR.rglob("*.png"))
        img_paths, msk_paths = [], []
        for p in all_png:
            parts_lower = [x.lower() for x in p.parts]
            if any(k in parts_lower for k in ["ground truth", "ground_truth", "masks", "mask"]):
                msk_paths.append(p)
            else:
                img_paths.append(p)
    else:
        img_paths = sorted(img_dir.glob("*.png"))
        msk_paths = sorted(msk_dir.glob("*.png"))

    print(f"[data] Found {len(img_paths)} images in: {img_dir}")
    print(f"[data] Found {len(msk_paths)} masks  in: {msk_dir}")

    # Match by filename stem (e.g., "1", "10", "100" ...)
    img_dict = {p.stem: p for p in img_paths}
    msk_dict = {p.stem: p for p in msk_paths}

    common = sorted(set(img_dict.keys()) & set(msk_dict.keys()), key=lambda x: int(x) if x.isdigit() else x)
    pairs  = [(img_dict[k], msk_dict[k]) for k in common]

    if len(pairs) == 0:
        print("[data] WARNING: stem-matching failed. Trying positional match...")
        min_n = min(len(img_paths), len(msk_paths))
        pairs = list(zip(sorted(img_paths)[:min_n], sorted(msk_paths)[:min_n]))

    print(f"[data] Matched {len(pairs)} image-mask pairs.")
    return pairs

# ─────────────────────────────────────────────
#  Step 4 — Preprocessing (identical to Kvasir)
# ─────────────────────────────────────────────
def preprocess_image(img_path: Path) -> torch.Tensor:
    """Load and preprocess a CVC image exactly as the Kvasir training pipeline."""
    image = Image.open(img_path).convert("RGB")
    image = TF.resize(image, IMAGE_SIZE, interpolation=TF.InterpolationMode.BILINEAR)
    image_tensor = TF.to_tensor(image)                      # [0,255] -> [0.0,1.0], C x H x W
    image_tensor = TF.normalize(image_tensor, MEAN, STD)   # ImageNet normalization
    return image_tensor                                      # shape: (3, 256, 256)


def preprocess_mask(msk_path: Path) -> np.ndarray:
    """Load and preprocess a CVC mask → binary numpy array (256×256)."""
    mask = Image.open(msk_path).convert("L")
    mask = TF.resize(mask, IMAGE_SIZE, interpolation=TF.InterpolationMode.NEAREST)
    mask_np = np.array(mask, dtype=np.float32) / 255.0
    mask_np = (mask_np > 0.5).astype(np.float32)           # ensure binary
    return mask_np                                          # shape: (256, 256)

# ─────────────────────────────────────────────
#  Step 5 — Metric computation
# ─────────────────────────────────────────────
def calculate_metrics(pred_prob: np.ndarray, target: np.ndarray, threshold: float = 0.5):
    pred_bin  = (pred_prob > threshold)
    target_bin = (target > 0.5)

    tp = np.sum(pred_bin & target_bin)
    fp = np.sum(pred_bin & ~target_bin)
    fn = np.sum(~pred_bin & target_bin)
    tn = np.sum(~pred_bin & ~target_bin)

    eps = 1e-7
    dice        = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou         = (tp + eps)       / (tp + fp + fn + eps)
    precision   = (tp + eps)       / (tp + fp + eps)
    recall      = (tp + eps)       / (tp + fn + eps)
    specificity = (tn + eps)       / (tn + fp + eps)

    return {
        "dice": float(dice), "iou": float(iou),
        "precision": float(precision), "recall": float(recall),
        "specificity": float(specificity)
    }

# ─────────────────────────────────────────────
#  Step 6 — Visualization
# ─────────────────────────────────────────────
def unnormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)


def save_visualization(img_tensor, gt_mask, pred_prob, metrics, save_path, img_name):
    """4-panel: Original | GT Mask | Predicted Mask | Overlay"""
    img_rgb  = unnormalize(img_tensor)
    img_uint8 = (img_rgb * 255).astype(np.uint8)

    pred_bin = (pred_prob > THRESHOLD).astype(np.uint8)

    # Green contour overlay
    overlay = img_uint8.copy()
    contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"{img_name}   |   Dice={metrics['dice']:.4f}   IoU={metrics['iou']:.4f}   "
        f"Prec={metrics['precision']:.4f}   Rec={metrics['recall']:.4f}   Spec={metrics['specificity']:.4f}",
        fontsize=9, y=1.01
    )

    axes[0].imshow(img_rgb);           axes[0].set_title("Original (CVC)");     axes[0].axis('off')
    axes[1].imshow(gt_mask, cmap='gray'); axes[1].set_title("Ground Truth");     axes[1].axis('off')
    axes[2].imshow(pred_bin, cmap='gray'); axes[2].set_title("Prediction (0.5)"); axes[2].axis('off')
    axes[3].imshow(overlay[:, :, ::-1]); axes[3].set_title("Boundary Overlay");  axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.close()

# ─────────────────────────────────────────────
#  Step 7 — Save prediction mask as PNG
# ─────────────────────────────────────────────
def save_prediction_mask(pred_prob: np.ndarray, save_path: Path):
    pred_bin_uint8 = (pred_prob > THRESHOLD).astype(np.uint8) * 255
    Image.fromarray(pred_bin_uint8, mode='L').save(save_path)

# ─────────────────────────────────────────────
#  Step 8 — Summary bar chart
# ─────────────────────────────────────────────
def save_summary_chart(summary: dict, save_path: Path):
    metrics  = ["dice", "iou", "precision", "recall", "specificity"]
    labels   = ["Dice", "IoU", "Precision", "Recall", "Specificity"]
    means    = [summary[f"mean_{m}"] for m in metrics]
    stds     = [summary[f"std_{m}"]  for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor='black')

    # Annotate bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.008,
                f"{mean:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("CVC-ClinicDB External Test — Attention U-Net (Kvasir-trained, 30ep)", fontsize=12)
    ax.axhline(y=0.75, color='red', linestyle='--', linewidth=1, label='Clinical threshold (0.75)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[viz] Saved summary chart -> {save_path}")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  CVC-ClinicDB External Generalization Test")
    print("  Model: Attention U-Net — Kvasir-trained (30 ep, no dropout)")
    print(f"  Device: {DEVICE}")
    print("="*65 + "\n")

    # ── Setup ──────────────────────────────────
    ensure_packages()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True,    exist_ok=True)
    PRED_DIR.mkdir(parents=True,   exist_ok=True)

    # ── Download dataset ───────────────────────
    download_cvc()

    # ── Discover pairs ─────────────────────────
    pairs = find_cvc_pairs()
    if len(pairs) == 0:
        print("[ERROR] No matched image-mask pairs found. Check dataset structure.")
        sys.exit(1)

    # ── Load model ─────────────────────────────
    print(f"\n[model] Loading checkpoint: {CHECKPOINT}")
    assert CHECKPOINT.exists(), f"Checkpoint not found: {CHECKPOINT}"

    from models.attention_unet import AttentionUNet
    model = AttentionUNet(in_channels=3, out_channels=1, dropout_rate=0.0)
    state = torch.load(CHECKPOINT, map_location=DEVICE)

    # Handle various checkpoint formats
    if isinstance(state, dict):
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Loaded. Parameters: {total_params:,}. Dropout=0 (inference mode).")

    # ── Inference loop ─────────────────────────
    all_metrics   = []
    per_image_out = []

    # Determine which indices to save as visualizations (evenly spread)
    n = len(pairs)
    viz_indices = set(np.linspace(0, n - 1, min(NUM_VIZ, n), dtype=int).tolist())

    print(f"\n[eval] Running inference on {n} CVC-ClinicDB pairs...\n")

    with torch.no_grad():
        for idx, (img_path, msk_path) in enumerate(tqdm(pairs, desc="Evaluating CVC")):
            img_tensor = preprocess_image(img_path)        # (3, 256, 256)
            gt_mask    = preprocess_mask(msk_path)          # (256, 256)

            # Forward pass
            input_batch = img_tensor.unsqueeze(0).to(DEVICE)   # (1, 3, 256, 256)
            logits = model(input_batch)                          # (1, 1, 256, 256)
            pred_prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (256, 256)

            # Metrics
            m = calculate_metrics(pred_prob, gt_mask, THRESHOLD)
            all_metrics.append(m)
            per_image_out.append({
                "image": img_path.name,
                "mask": msk_path.name,
                **m
            })

            # Save prediction PNG
            pred_save = PRED_DIR / f"pred_{img_path.stem}.png"
            save_prediction_mask(pred_prob, pred_save)

            # Save visualization for sampled indices
            if idx in viz_indices:
                viz_save = VIZ_DIR / f"viz_{idx:04d}_{img_path.stem}.png"
                save_visualization(img_tensor, gt_mask, pred_prob, m, viz_save, img_path.stem)

    # ── Compute summary ─────────────────────────
    metric_keys = ["dice", "iou", "precision", "recall", "specificity"]
    summary = {}
    for key in metric_keys:
        vals = [m[key] for m in all_metrics]
        summary[f"mean_{key}"] = float(np.mean(vals))
        summary[f"std_{key}"]  = float(np.std(vals))

    # ── Print results ───────────────────────────
    print("\n" + "="*65)
    print("  CVC-ClinicDB External Test — RESULTS")
    print("="*65)
    print(f"  Total images evaluated : {n}")
    print(f"  Checkpoint             : review_1_best.pth (Kvasir-trained, 30ep)")
    print(f"  Preprocessing          : 256x256, ImageNet norm, threshold=0.5")
    print()
    for key in metric_keys:
        print(f"  {key.capitalize():<15s}: {summary[f'mean_{key}']:.4f} +/- {summary[f'std_{key}']:.4f}")
    print("="*65 + "\n")

    # ── Save metrics.json ────────────────────────
    metrics_dict = {
        "experiment": "CVC-ClinicDB External Generalization Test",
        "model": "Attention U-Net (Kvasir-trained, 30ep, no dropout)",
        "checkpoint": str(CHECKPOINT),
        "dataset": "CVC-ClinicDB (balraj98/cvcclinicdb on Kaggle)",
        "num_images": n,
        "image_size": list(IMAGE_SIZE),
        "threshold": THRESHOLD,
        "normalization": {"mean": MEAN, "std": STD},
        "summary": summary,
        "per_image": per_image_out
    }
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[out] Saved metrics.json -> {metrics_path}")

    # ── Save summary bar chart ───────────────────
    save_summary_chart(summary, OUTPUT_DIR / "cvc_summary_chart.png")

    # ── Save a compact comparison table ─────────
    kvasir_metrics = {
        "mean_dice": 0.8120, "std_dice": 0.2282,
        "mean_iou": 0.7307,  "std_iou": 0.2506,
        "mean_precision": 0.8532, "std_precision": 0.2121,
        "mean_recall": 0.8520,   "std_recall": 0.2329,
        "mean_specificity": 0.9733, "std_specificity": 0.0485,
    }
    comparison = {
        "Kvasir-SEG (in-domain, 30ep)": kvasir_metrics,
        "CVC-ClinicDB (external, zero-shot)": summary
    }
    comparison_path = OUTPUT_DIR / "kvasir_vs_cvc_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=4)
    print(f"[out] Saved comparison -> {comparison_path}")

    # ── Save comparison chart ────────────────────
    save_comparison_chart(kvasir_metrics, summary, OUTPUT_DIR / "kvasir_vs_cvc_chart.png")

    print(f"\n[done] All outputs saved to: {OUTPUT_DIR}")
    print(f"       Predictions : {PRED_DIR}  ({n} masks)")
    print(f"       Samples     : {VIZ_DIR}   ({min(NUM_VIZ, n)} visualizations)")
    print()


def save_comparison_chart(kvasir: dict, cvc: dict, save_path: Path):
    """Side-by-side grouped bar chart comparing Kvasir vs CVC results."""
    metrics = ["dice", "iou", "precision", "recall", "specificity"]
    labels  = ["Dice", "IoU", "Precision", "Recall", "Specificity"]

    kvasir_means = [kvasir[f"mean_{m}"] for m in metrics]
    kvasir_stds  = [kvasir[f"std_{m}"]  for m in metrics]
    cvc_means    = [cvc[f"mean_{m}"]    for m in metrics]
    cvc_stds     = [cvc[f"std_{m}"]     for m in metrics]

    x    = np.arange(len(labels))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))

    b1 = ax.bar(x - w/2, kvasir_means, w, yerr=kvasir_stds, label="Kvasir-SEG (in-domain)",
                capsize=5, color="#4C72B0", alpha=0.85, edgecolor='black')
    b2 = ax.bar(x + w/2, cvc_means,    w, yerr=cvc_stds,    label="CVC-ClinicDB (external/zero-shot)",
                capsize=5, color="#C44E52", alpha=0.85, edgecolor='black')

    for bar, val in zip(b1, kvasir_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#4C72B0')
    for bar, val in zip(b2, cvc_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#C44E52')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Attention U-Net: Kvasir-SEG (In-Domain) vs CVC-ClinicDB (External Test)",
                 fontsize=11, fontweight='bold')
    ax.axhline(y=0.75, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Clinical threshold (0.75)')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[viz] Saved comparison chart -> {save_path}")


if __name__ == "__main__":
    main()
