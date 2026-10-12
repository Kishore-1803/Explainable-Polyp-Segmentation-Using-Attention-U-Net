"""
Diagnoses WHY the zero-shot CVC-ClinicDB score stays low, and tests whether a
different metric-aggregation convention could explain the base paper's figures.

    python experiments/paper_replication/diagnose_cvc.py --arm paper

Reports, for both Kvasir-test and CVC:
  * per-image mean Dice / IoU      (our convention, and the standard one)
  * micro-averaged Dice / IoU      (TP/FP/FN pooled over the whole dataset)
  * failure profile                (complete misses, empty predictions)
  * Dice vs. image brightness      (is the failure illumination-linked?)
  * the IoU/Dice ratio, compared against the base paper's reported ratio.

    NOTE on the ratio: for a single image IoU = Dice/(2-Dice) <= Dice, so
    mean IoU <= mean Dice always. That is the only hard bound. Because
    d -> d/(2-d) is CONVEX, Jensen's inequality gives
        mean IoU >= mean_Dice / (2 - mean_Dice),
    i.e. f(mean Dice) is a LOWER bound on mean IoU, not an upper one. A high
    IoU/Dice ratio is therefore unusual but not impossible: it indicates a
    strongly bimodal score distribution (near-perfect hits plus outright
    misses). It is evidence about the score distribution, NOT proof of error.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from paper_model import PaperAttentionUNet      # noqa: E402

CACHE = ROOT / "_cache"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@torch.no_grad()
def collect(model, device, images, masks, idx, batch=16, label=""):
    """Returns per-image stats plus pooled TP/FP/FN."""
    rows = []
    TP = FP = FN = 0
    for s in tqdm(range(0, len(idx), batch), desc=f"diag:{label}"):
        chunk = idx[s:s + batch]
        arr = np.stack([np.asarray(images[j]) for j in chunk])
        x = torch.from_numpy(arr.transpose(0, 3, 1, 2).copy()).float().div_(255.0)
        x = ((x - MEAN) / STD).to(device)
        prob = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
        for k, j in enumerate(chunk):
            pred = prob[k] > 0.5
            gt = np.asarray(masks[j]).astype(bool)
            tp = int(np.sum(pred & gt)); fp = int(np.sum(pred & ~gt)); fn = int(np.sum(~pred & gt))
            TP += tp; FP += fp; FN += fn
            eps = 1e-7
            rows.append({
                "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
                "iou": (tp + eps) / (tp + fp + fn + eps),
                "pred_px": int(pred.sum()),
                "gt_px": int(gt.sum()),
                "brightness": float(arr[k].mean() / 255.0),
            })
    eps = 1e-7
    micro = {
        "dice": (2 * TP + eps) / (2 * TP + FP + FN + eps),
        "iou": (TP + eps) / (TP + FP + FN + eps),
    }
    return rows, micro


def report(label, rows, micro):
    d = np.array([r["dice"] for r in rows])
    i = np.array([r["iou"] for r in rows])
    pred = np.array([r["pred_px"] for r in rows])
    gt = np.array([r["gt_px"] for r in rows])
    br = np.array([r["brightness"] for r in rows])

    print(f"\n{'='*66}\n{label}  (n={len(rows)})\n{'='*66}")
    print(f"  per-image mean   Dice {d.mean():.4f}   IoU {i.mean():.4f}")
    print(f"  micro-averaged   Dice {micro['dice']:.4f}   IoU {micro['iou']:.4f}")
    print(f"  median           Dice {np.median(d):.4f}")
    print(f"\n  failure profile:")
    print(f"    Dice < 0.05 (complete miss) : {int((d < 0.05).sum()):4d}  ({(d<0.05).mean()*100:.1f}%)")
    print(f"    Dice > 0.80 (good)          : {int((d > 0.80).sum()):4d}  ({(d>0.80).mean()*100:.1f}%)")
    print(f"    empty prediction            : {int((pred == 0).sum()):4d}")
    print(f"    over-prediction (pred>2x gt): {int((pred > 2 * np.maximum(gt,1)).sum()):4d}")
    print(f"\n  mean predicted area {pred.mean():8.0f} px   mean GT area {gt.mean():8.0f} px"
          f"   ratio {pred.mean()/max(gt.mean(),1):.2f}x")

    lo, hi = br < np.median(br), br >= np.median(br)
    print(f"\n  Dice by image brightness:")
    print(f"    darker  half  Dice {d[lo].mean():.4f}")
    print(f"    brighter half Dice {d[hi].mean():.4f}")
    if len(d) > 2:
        print(f"    corr(Dice, brightness) = {np.corrcoef(d, br)[0,1]:+.3f}")

    floor = d.mean() / (2 - d.mean())
    print(f"\n  IoU/Dice ratio = {i.mean()/max(d.mean(),1e-9):.3f}"
          f"   (Jensen lower bound on mean IoU = {floor:.4f}; hard upper bound"
          f" = mean Dice = {d.mean():.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["paper", "ours"], default="paper")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaperAttentionUNet(base=64).to(device)
    model.load_state_dict(torch.load(
        ROOT / "checkpoints" / f"paper_repl_{args.arm}" / "best.pth", map_location=device))
    model.eval()

    manifest = json.loads((CACHE / "manifest.json").read_text())
    order, assign = manifest["kvasir_order"], manifest["kvasir_split"]
    kidx = [i for i, fn in enumerate(order) if assign[fn] == "test"]

    ki = np.load(CACHE / f"kvasir_{args.arm}_images.npy", mmap_mode="r")
    km = np.load(CACHE / "kvasir_masks.npy", mmap_mode="r")
    rows, micro = collect(model, device, ki, km, kidx, label="kvasir")
    report(f"KVASIR-SEG test  [arm={args.arm}]", rows, micro)

    ci = np.load(CACHE / f"cvc_{args.arm}_images.npy", mmap_mode="r")
    cm = np.load(CACHE / "cvc_masks.npy", mmap_mode="r")
    crows, cmicro = collect(model, device, ci, cm, list(range(len(ci))), label="cvc")
    report(f"CVC-ClinicDB zero-shot  [arm={args.arm}]", crows, cmicro)

    print(f"\n{'='*66}")
    print("Base paper reports:  Kvasir DSC 0.8012 / IoU 0.7878  (ratio 0.983)")
    print("                     CVC    DSC 0.7823 / IoU 0.7654  (ratio 0.978)")
    print("Both satisfy the hard bound mean IoU <= mean Dice, so they are NOT")
    print("mathematically impossible. They are, however, far tighter than ours")
    print("and imply a strongly bimodal distribution: e.g. ~80% of frames at")
    print("Dice ~0.98 and ~20% at ~0 reproduces their CVC pair almost exactly.")
    print("Near-perfect agreement on 80% of unseen-device frames is an unusual")
    print("profile; it may equally indicate a different metric convention.")
    print("We report this as an open discrepancy, not as an error.")
    print("=" * 66)


if __name__ == "__main__":
    main()
