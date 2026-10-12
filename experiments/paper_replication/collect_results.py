"""
Collects every run into one comparison table and emits LaTeX for the report
and the slide deck.

    python experiments/paper_replication/collect_results.py

Rows: the base paper's reported figures, our original 7.85M model, the
full-width replication, and each architecture/loss variant.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "results"

# Reported by Gupta & Benila (IEEE Access 2025). Validation means, 5-fold CV.
PAPER = {
    "label": "Gupta \\& Benila (reported)",
    "kvasir": {"dice": 0.8012, "iou": 0.7878},
    "cvc": {"dice": 0.7823, "iou": 0.7654},
    "params": 31_200_000,
    "note": "validation, 5-fold",
}

# Our original half-width pipeline (EXP-01 / EXP-08), from the earlier study.
ORIGINAL = [
    ("Our original Attention U-Net (7.85M)", 0.8120, 0.7307, 0.4516, 0.3495, 7_851_773),
    ("Our original + L* CLAHE (EXP-08)", 0.7953, 0.7106, 0.6568, 0.5676, 7_851_773),
]

RUNS = [
    ("paper_repl_paper", "Replication: paper preprocessing"),
    ("paper_repl_ours", "Replication + L* CLAHE preprocessing"),
    ("paper_var_se", "+ SE channel attention"),
    ("paper_var_ds", "+ Deep supervision"),
    ("paper_var_se_ds", "+ SE + deep supervision"),
    ("paper_var_baseline_tversky", "+ Tversky loss ($\\alpha$=0.7)"),
]


def load(run):
    p = RES / run / "metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    k, c = d["kvasir_test"]["summary"], d["cvc_zero_shot"]["summary"]
    cfg = RES / run / "train_config.json"
    params = json.loads(cfg.read_text()).get("params") if cfg.exists() else None
    return {
        "kvasir_dice": k["mean_dice"], "kvasir_iou": k["mean_iou"],
        "cvc_dice": c["mean_dice"], "cvc_iou": c["mean_iou"],
        "cvc_prec": c["mean_precision"], "cvc_rec": c["mean_recall"],
        "area": c.get("area_ratio"), "params": params,
    }


def main():
    rows = []
    rows.append((PAPER["label"], PAPER["kvasir"]["dice"], PAPER["kvasir"]["iou"],
                 PAPER["cvc"]["dice"], PAPER["cvc"]["iou"], None, None, None,
                 PAPER["params"]))
    for lbl, kd, ki, cd, ci, pr in ORIGINAL:
        rows.append((lbl, kd, ki, cd, ci, None, None, None, pr))

    found = {}
    for run, label in RUNS:
        r = load(run)
        if r is None:
            print(f"  (missing: {run})", file=sys.stderr)
            continue
        found[run] = r
        rows.append((label, r["kvasir_dice"], r["kvasir_iou"], r["cvc_dice"],
                     r["cvc_iou"], r["cvc_prec"], r["cvc_rec"], r["area"], r["params"]))

    w = 44
    print(f"\n{'Model':{w}s} {'Kv Dice':>8s} {'Kv IoU':>7s} {'CVC Dice':>9s} "
          f"{'CVC IoU':>8s} {'Prec':>6s} {'Rec':>6s} {'Area':>6s} {'Params':>12s}")
    print("-" * (w + 70))
    for lbl, kd, ki, cd, ci, pr, rc, ar, pm in rows:
        clean = lbl.replace("\\&", "&").replace("$\\alpha$", "alpha").replace("=0.7)", "=0.7)")
        print(f"{clean:{w}.{w}s} {kd:8.4f} {ki:7.4f} {cd:9.4f} {ci:8.4f} "
              f"{(f'{pr:.4f}' if pr else '   -  '):>6s} "
              f"{(f'{rc:.4f}' if rc else '   -  '):>6s} "
              f"{(f'{ar:.2f}x' if ar else '  -  '):>6s} "
              f"{(f'{pm:,}' if pm else '-'):>12s}")

    # Best variant selected by Kvasir validation Dice (never by CVC).
    variants = {k: v for k, v in found.items() if k.startswith("paper_var_")}
    if variants:
        best = max(variants, key=lambda k: json.loads(
            (RES / k / "train_config.json").read_text())["best_val_dice"])
        bv = json.loads((RES / best / "train_config.json").read_text())["best_val_dice"]
        print(f"\nBest variant by Kvasir VALIDATION Dice: {best}  (val {bv:.4f})")
        print(f"  its Kvasir test Dice {found[best]['kvasir_dice']:.4f}, "
              f"CVC {found[best]['cvc_dice']:.4f}, area {found[best]['area']:.2f}x")

    json.dump({k: v for k, v in found.items()},
              open(RES / "ablation_summary.json", "w"), indent=2)
    print(f"\nSaved {RES / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
