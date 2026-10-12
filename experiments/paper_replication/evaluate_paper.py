"""
Evaluates a trained paper-replication checkpoint on both datasets.

    python experiments/paper_replication/evaluate_paper.py --arm paper

Reports mean +/- std Dice, IoU, Precision, Recall and Specificity at tau = 0.5
on the Kvasir-SEG held-out test split (in-domain) and on all 612 CVC-ClinicDB
frames (zero-shot). Uses the same per-image metric routine as every other
experiment in this project (src/evaluation/evaluator.calculate_metrics), so the
numbers are directly comparable with our earlier runs.
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

from paper_model import PaperAttentionUNet              # noqa: E402
from evaluation.evaluator import calculate_metrics      # noqa: E402

CACHE = ROOT / "_cache"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
KEYS = ["dice", "iou", "precision", "recall", "specificity"]


def load_set(tag, arm, which=None):
    images = np.load(CACHE / f"{tag}_{arm}_images.npy", mmap_mode="r")
    masks = np.load(CACHE / f"{tag}_masks.npy", mmap_mode="r")
    if which is None:
        return images, masks, list(range(len(images)))
    manifest = json.loads((CACHE / "manifest.json").read_text())
    order, assign = manifest["kvasir_order"], manifest["kvasir_split"]
    idx = [i for i, fn in enumerate(order) if assign[fn] == which]
    return images, masks, idx


@torch.no_grad()
def run(model, device, images, masks, idx, batch=16, label=""):
    model.eval()
    per_image = []
    for s in tqdm(range(0, len(idx), batch), desc=f"eval:{label}"):
        chunk = idx[s:s + batch]
        arr = np.stack([np.asarray(images[j]) for j in chunk])
        x = torch.from_numpy(arr.transpose(0, 3, 1, 2).copy()).float().div_(255.0)
        x = ((x - MEAN) / STD).to(device)
        probs = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
        for k, j in enumerate(chunk):
            per_image.append(calculate_metrics(probs[k], np.asarray(masks[j]).astype(np.float32)))

    summary = {}
    for key in KEYS:
        vals = [m[key] for m in per_image]
        summary[f"mean_{key}"] = float(np.mean(vals))
        summary[f"std_{key}"] = float(np.std(vals))
    return {"summary": summary, "per_image": per_image}


def show(label, s):
    print(f"\n--- {label} ---")
    for k in KEYS:
        print(f"  {k.capitalize():12s} {s[f'mean_{k}']:.4f} +/- {s[f'std_{k}']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["paper", "ours"], required=True)
    ap.add_argument("--base", type=int, default=64)
    args = ap.parse_args()

    run_name = f"paper_repl_{args.arm}"
    ckpt = ROOT / "checkpoints" / run_name / "best.pth"
    res_dir = ROOT / "results" / run_name
    res_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaperAttentionUNet(base=args.base).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded {ckpt}")

    ki, km, kidx = load_set("kvasir", args.arm, "test")
    kv = run(model, device, ki, km, kidx, label="kvasir-test")

    ci, cm, cidx = load_set("cvc", args.arm)
    cvc = run(model, device, ci, cm, cidx, label="cvc-all")

    show(f"Kvasir-SEG test ({len(kidx)} images, arm={args.arm})", kv["summary"])
    show(f"CVC-ClinicDB zero-shot ({len(cidx)} images, arm={args.arm})", cvc["summary"])

    out = {"arm": args.arm, "checkpoint": str(ckpt),
           "kvasir_test": kv, "cvc_zero_shot": cvc}
    with open(res_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {res_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
