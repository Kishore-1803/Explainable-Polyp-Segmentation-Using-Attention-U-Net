"""
Evaluates a trained variant on Kvasir-SEG test and CVC-ClinicDB zero-shot.

    python experiments/paper_replication/evaluate_variant.py --variant se

Also reports the predicted/ground-truth area ratio, which is the quantity the
variants are meant to fix (the replicated paper model over-segments CVC by
2.18x).
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

from variants import build_model                       # noqa: E402
from evaluation.evaluator import calculate_metrics     # noqa: E402

CACHE = ROOT / "_cache"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
KEYS = ["dice", "iou", "precision", "recall", "specificity"]


@torch.no_grad()
def evaluate(model, device, images, masks, idx, label, batch=16):
    rows, pred_px, gt_px = [], [], []
    for s in tqdm(range(0, len(idx), batch), desc=f"eval:{label}"):
        chunk = idx[s:s + batch]
        arr = np.stack([np.asarray(images[j]) for j in chunk])
        x = torch.from_numpy(arr.transpose(0, 3, 1, 2).copy()).float().div_(255.0)
        x = ((x - MEAN) / STD).to(device)
        prob = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
        for k, j in enumerate(chunk):
            gt = np.asarray(masks[j]).astype(np.float32)
            rows.append(calculate_metrics(prob[k], gt))
            pred_px.append(int((prob[k] > 0.5).sum()))
            gt_px.append(int(gt.sum()))

    summary = {}
    for key in KEYS:
        vals = [m[key] for m in rows]
        summary[f"mean_{key}"] = float(np.mean(vals))
        summary[f"std_{key}"] = float(np.std(vals))
    d = np.array([r["dice"] for r in rows])
    summary["area_ratio"] = float(np.mean(pred_px) / max(np.mean(gt_px), 1))
    summary["complete_miss"] = int((d < 0.05).sum())
    summary["good"] = int((d > 0.80).sum())
    summary["n"] = len(rows)
    return {"summary": summary, "per_image": rows}


def show(label, s):
    print(f"\n--- {label} ---")
    for k in KEYS:
        print(f"  {k.capitalize():12s} {s[f'mean_{k}']:.4f} +/- {s[f'std_{k}']:.4f}")
    print(f"  {'Area ratio':12s} {s['area_ratio']:.2f}x   "
          f"misses {s['complete_miss']}/{s['n']}   good(>0.8) {s['good']}/{s['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "se", "ds", "se_ds"], required=True)
    ap.add_argument("--loss", choices=["bcedice", "tversky"], default="bcedice")
    ap.add_argument("--arm", default="paper", choices=["paper", "ours"])
    ap.add_argument("--base", type=int, default=64)
    args = ap.parse_args()

    tag = args.variant if args.loss == "bcedice" else f"{args.variant}_{args.loss}"
    if args.arm != "paper":
        tag = f"{tag}_{args.arm}"
    run = f"paper_var_{tag}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.variant, base=args.base).to(device)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / run / "best.pth",
                                     map_location=device))
    model.eval()

    manifest = json.loads((CACHE / "manifest.json").read_text())
    order, assign = manifest["kvasir_order"], manifest["kvasir_split"]
    kidx = [i for i, fn in enumerate(order) if assign[fn] == "test"]

    ki = np.load(CACHE / f"kvasir_{args.arm}_images.npy", mmap_mode="r")
    km = np.load(CACHE / "kvasir_masks.npy", mmap_mode="r")
    kv = evaluate(model, device, ki, km, kidx, "kvasir")

    ci = np.load(CACHE / f"cvc_{args.arm}_images.npy", mmap_mode="r")
    cm = np.load(CACHE / "cvc_masks.npy", mmap_mode="r")
    cvc = evaluate(model, device, ci, cm, list(range(len(ci))), "cvc")

    show(f"Kvasir-SEG test   [{tag}]", kv["summary"])
    show(f"CVC zero-shot     [{tag}]", cvc["summary"])

    out = {"variant": args.variant, "loss": args.loss, "arm": args.arm,
           "kvasir_test": kv, "cvc_zero_shot": cvc}
    json.dump(out, open(ROOT / "results" / run / "metrics.json", "w"), indent=2)
    print(f"\nSaved results/{run}/metrics.json")


if __name__ == "__main__":
    main()
