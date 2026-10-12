"""
Regenerates every downstream result for the chosen replication-study model, so
that the report and slides no longer depend on the earlier 7.85M experiments.

    python experiments/paper_replication/regenerate_figures.py --variant se_ds

Produces, under results/paper_var_<tag>/ and the report/deck figure folders:
  1. qualitative_samples.png  -- 3 Kvasir test cases: input, GT, prediction
  2. gradcam_samples.png      -- 2 cases: input, GT, prediction, Grad-CAM
  3. gradcam_quantification.json -- mean/median localisation Dice over the
                                    Kvasir test split (Otsu-thresholded CAM)
  4. threshold_sweep.png + threshold_sweep.json -- CVC sweep tau in [0.1, 0.5]
  5. clahe_comparison.png     -- one CVC frame under both preprocessing arms

Grad-CAM target layer is `up4`, the deepest decoder up-convolution, matching
the definition used throughout the report.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from variants import build_model                      # noqa: E402
from explainability.grad_cam import SegmentationGradCAM  # noqa: E402

CACHE = ROOT / "_cache"
REPORT_FIG = ROOT.parent / "ieee_report" / "ieee_report" / "figures"
DECK_FIG = ROOT.parent / "presentation_overleaf" / "figures"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_tensor(arr_u8):
    x = torch.from_numpy(arr_u8.transpose(2, 0, 1).copy()).float().div_(255.0)
    return ((x - MEAN) / STD).unsqueeze(0)


def dice(a, b, eps=1e-7):
    a, b = a.astype(bool), b.astype(bool)
    return (2 * (a & b).sum() + eps) / (a.sum() + b.sum() + eps)


# ---------------------------------------------------------------- figure 1
def qualitative(model, device, imgs, masks, idx, out_path, n=3):
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.5 * n))
    titles = ["Original Image", "Ground Truth Mask", "Predicted Mask ($\\tau$=0.5)"]
    for r, j in enumerate(idx[:n]):
        img = np.asarray(imgs[j])
        gt = np.asarray(masks[j])
        with torch.no_grad():
            pred = torch.sigmoid(model(to_tensor(img).to(device)))[0, 0].cpu().numpy() > 0.5
        for c, (panel, cmap) in enumerate([(img, None), (gt, "gray"), (pred, "gray")]):
            ax = axes[r, c]
            ax.imshow(panel, cmap=cmap)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
        axes[r, 2].text(0.02, 0.04, f"Dice {dice(pred, gt):.3f}", color="lime",
                        fontsize=9, transform=axes[r, 2].transAxes, weight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def gradcam_figure(model, device, imgs, masks, idx, out_path, n=2):
    cam_tool = SegmentationGradCAM(model, model.up4)
    fig, axes = plt.subplots(n, 4, figsize=(9.6, 2.5 * n))
    titles = ["Original Image", "Ground Truth", "Predicted Mask",
              "Grad-CAM (decoder up4)"]
    for r, j in enumerate(idx[:n]):
        img = np.asarray(imgs[j])
        gt = np.asarray(masks[j])
        x = to_tensor(img).to(device)
        cam, logits = cam_tool.generate_heatmap(x)
        cam = np.squeeze(np.asarray(cam))
        if cam.shape != (256, 256):
            cam = cv2.resize(cam, (256, 256))
        cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
        pred = torch.sigmoid(logits.detach())[0, 0].cpu().numpy() > 0.5
        for c in range(4):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(img)
            elif c == 1:
                ax.imshow(gt, cmap="gray")
            elif c == 2:
                ax.imshow(pred, cmap="gray")
            else:
                ax.imshow(img)
                ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    cam_tool.remove_hooks()


# ---------------------------------------------------------------- metric 3
def gradcam_localisation(model, device, imgs, masks, idx):
    cam_tool = SegmentationGradCAM(model, model.up4)
    scores = []
    for j in tqdm(idx, desc="gradcam-dice"):
        img, gt = np.asarray(imgs[j]), np.asarray(masks[j])
        cam, _ = cam_tool.generate_heatmap(to_tensor(img).to(device))
        cam = np.squeeze(np.asarray(cam))
        if cam.shape != (256, 256):
            cam = cv2.resize(cam, (256, 256))
        cam8 = ((cam - cam.min()) / (np.ptp(cam) + 1e-8) * 255).astype(np.uint8)
        _, binm = cv2.threshold(cam8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scores.append(float(dice(binm > 0, gt > 0)))
    cam_tool.remove_hooks()
    return {"mean_dice": float(np.mean(scores)),
            "median_dice": float(np.median(scores)),
            "std_dice": float(np.std(scores)), "n": len(scores)}


# ---------------------------------------------------------------- figure 4
@torch.no_grad()
def threshold_sweep(model, device, imgs, masks, taus=(0.1, 0.2, 0.3, 0.4, 0.5)):
    probs = []
    for s in tqdm(range(0, len(imgs), 16), desc="threshold-sweep"):
        batch = np.stack([np.asarray(imgs[j]) for j in range(s, min(s + 16, len(imgs)))])
        x = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy()).float().div_(255.0)
        x = ((x - MEAN) / STD).to(device)
        probs.append(torch.sigmoid(model(x))[:, 0].cpu().numpy())
    probs = np.concatenate(probs)
    gts = np.stack([np.asarray(masks[j]) for j in range(len(masks))]).astype(bool)

    out = {}
    eps = 1e-7
    for t in taus:
        d, i, p, r = [], [], [], []
        for k in range(len(probs)):
            pr, gt = probs[k] > t, gts[k]
            tp = (pr & gt).sum(); fp = (pr & ~gt).sum(); fn = (~pr & gt).sum()
            d.append((2 * tp + eps) / (2 * tp + fp + fn + eps))
            i.append((tp + eps) / (tp + fp + fn + eps))
            p.append((tp + eps) / (tp + fp + eps))
            r.append((tp + eps) / (tp + fn + eps))
        out[f"{t:.1f}"] = {"dice": float(np.mean(d)), "iou": float(np.mean(i)),
                           "precision": float(np.mean(p)), "recall": float(np.mean(r))}
    return out


def plot_sweep(sweep, out_path):
    ts = [float(t) for t in sweep]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for key, colour, marker in [("dice", "#2F6DB5", "o"), ("iou", "#4E7F6B", "s"),
                                ("precision", "#D1782F", "^"), ("recall", "#C2253F", "v")]:
        ax.plot(ts, [sweep[t][key] for t in sweep], marker=marker, color=colour,
                label=key.capitalize(), lw=1.8)
    ax.set_xlabel("Decision threshold $\\tau$")
    ax.set_ylabel("Score")
    ax.set_title("CVC-ClinicDB threshold sensitivity (612 frames)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def preprocessing_comparison(out_path, frame=2):
    paper = np.load(CACHE / "cvc_paper_images.npy", mmap_mode="r")
    ours = np.load(CACHE / "cvc_ours_images.npy", mmap_mode="r")
    masks = np.load(CACHE / "cvc_masks.npy", mmap_mode="r")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
    for ax, (panel, title) in zip(axes, [
            (np.asarray(masks[frame]), "Ground truth"),
            (np.asarray(paper[frame]), "Paper preprocessing\n(min-max + hist-eq + CLAHE + blur)"),
            (np.asarray(ours[frame]), "Our preprocessing\n($L^*$-only CLAHE)")]):
        ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="se_ds")
    ap.add_argument("--arm", default="paper")
    args = ap.parse_args()

    tag = args.variant if args.arm == "paper" else f"{args.variant}_{args.arm}"
    run = f"paper_var_{tag}"
    res = ROOT / "results" / run
    res.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.variant, base=64).to(device)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / run / "best.pth",
                                     map_location=device))
    model.eval()
    print(f"Loaded {run}")

    manifest = json.loads((CACHE / "manifest.json").read_text())
    order, assign = manifest["kvasir_order"], manifest["kvasir_split"]
    kidx = [i for i, fn in enumerate(order) if assign[fn] == "test"]
    ki = np.load(CACHE / f"kvasir_{args.arm}_images.npy", mmap_mode="r")
    km = np.load(CACHE / "kvasir_masks.npy", mmap_mode="r")

    qualitative(model, device, ki, km, kidx, res / "qualitative_samples.png")
    print("  qualitative_samples.png")

    gradcam_figure(model, device, ki, km, kidx, res / "gradcam_samples.png")
    print("  gradcam_samples.png")

    q = gradcam_localisation(model, device, ki, km, kidx)
    json.dump(q, open(res / "gradcam_quantification.json", "w"), indent=2)
    print(f"  Grad-CAM localisation Dice: {q['mean_dice']:.4f} "
          f"+/- {q['std_dice']:.4f} (median {q['median_dice']:.4f})")

    ci = np.load(CACHE / f"cvc_{args.arm}_images.npy", mmap_mode="r")
    cm = np.load(CACHE / "cvc_masks.npy", mmap_mode="r")
    sweep = threshold_sweep(model, device, ci, cm)
    json.dump(sweep, open(res / "threshold_sweep.json", "w"), indent=2)
    plot_sweep(sweep, res / "threshold_sweep.png")
    print("  threshold_sweep.png")
    for t, v in sweep.items():
        print(f"    tau={t}  Dice {v['dice']:.4f}  IoU {v['iou']:.4f}  "
              f"P {v['precision']:.4f}  R {v['recall']:.4f}")

    preprocessing_comparison(res / "preprocessing_comparison.png")
    print("  preprocessing_comparison.png")

    for name, dest in [("qualitative_samples.png", "qualitative_samples.png"),
                       ("gradcam_samples.png", "gradcam_samples.png"),
                       ("threshold_sweep.png", "threshold_analysis.png"),
                       ("preprocessing_comparison.png", "preprocessing_comparison.png")]:
        for folder in (REPORT_FIG, DECK_FIG):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / dest).write_bytes((res / name).read_bytes())
    print(f"\nCopied figures into {REPORT_FIG} and {DECK_FIG}")


if __name__ == "__main__":
    main()
