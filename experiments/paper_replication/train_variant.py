"""
Trains an architecture/loss variant on the BASE PAPER's preprocessing arm.

    python experiments/paper_replication/train_variant.py --variant se
    python experiments/paper_replication/train_variant.py --variant ds
    python experiments/paper_replication/train_variant.py --variant baseline --loss tversky

Everything except the named change is held fixed and identical to the
replication run: paper preprocessing, paper augmentation, 31.4M-parameter
backbone, batch 16, Adam 1e-4, cosine schedule, early stopping patience 10,
seed 42. Only the architecture (or the loss) differs, so any delta is
attributable to that change.

MODEL SELECTION: the checkpoint is chosen by best KVASIR VALIDATION Dice.
CVC-ClinicDB is never used for selection -- it stays a held-out zero-shot set.
All variants are reported, not only the best one.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from variants import build_model, build_loss          # noqa: E402
from train_paper import CachedKvasir, batch_metrics   # noqa: E402

AUX_WEIGHTS = (0.4, 0.3, 0.2)   # deep-supervision aux heads: d2, d3, d4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "se", "ds", "se_ds"], required=True)
    ap.add_argument("--loss", choices=["bcedice", "tversky"], default="bcedice")
    ap.add_argument("--arm", default="paper", choices=["paper", "ours"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = args.variant if args.loss == "bcedice" else f"{args.variant}_{args.loss}"
    if args.arm != "paper":                 # keep arms in separate run folders
        tag = f"{tag}_{args.arm}"
    run = f"paper_var_{tag}"
    ckpt_dir = ROOT / "checkpoints" / run
    res_dir = ROOT / "results" / run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(ckpt_dir / "train.log", "a") as f:
            f.write(line + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    train_ds = CachedKvasir(args.arm, "train", augment=True, seed=args.seed)
    val_ds = CachedKvasir(args.arm, "val", augment=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=amp)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=amp)

    model = build_model(args.variant, base=args.base).to(device)
    criterion = build_loss(args.loss)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    log(f"variant={args.variant} loss={args.loss} arm={args.arm} "
        f"params={n_params:,} device={device} amp={amp}")

    history = {k: [] for k in ("train_loss", "train_dice", "val_loss", "val_dice", "val_iou")}
    best_dice, best_loss, stale = 0.0, float("inf"), 0

    for epoch in range(args.epochs):
        model.train()
        agg, n = np.zeros(2), 0
        for x, y in tqdm(train_ld, desc=f"[{tag}] {epoch+1}/{args.epochs} train"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(x)
                if isinstance(out, tuple):
                    logits, aux = out
                    loss = criterion(logits, y)
                    for w, a in zip(AUX_WEIGHTS, aux):
                        loss = loss + w * criterion(a, y)
                else:
                    logits = out
                    loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            d, _ = batch_metrics(logits.float(), y)
            agg += np.array([loss.item(), d]) * x.size(0)
            n += x.size(0)
        tr_loss, tr_dice = agg / n

        model.eval()
        agg, n = np.zeros(3), 0
        with torch.no_grad():
            for x, y in tqdm(val_ld, desc=f"[{tag}] {epoch+1}/{args.epochs} val"):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                d, i = batch_metrics(logits.float(), y)
                agg += np.array([loss.item(), d, i]) * x.size(0)
                n += x.size(0)
        va_loss, va_dice, va_iou = agg / n

        scheduler.step()
        for k, v in zip(history, (tr_loss, tr_dice, va_loss, va_dice, va_iou)):
            history[k].append(float(v))

        log(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss {tr_loss:.4f} Dice {tr_dice:.4f}"
            f" || Val Loss {va_loss:.4f} Dice {va_dice:.4f} IoU {va_iou:.4f}")

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            log(f"  -> new best val Dice {best_dice:.4f}")

        if va_loss < best_loss - 1e-5:
            best_loss, stale = va_loss, 0
        else:
            stale += 1
            if stale >= args.patience:
                log(f"Early stopping after {epoch+1} epochs.")
                break

    json.dump(history, open(res_dir / "training_history.json", "w"), indent=2)
    json.dump({**vars(args), "params": n_params, "best_val_dice": best_dice},
              open(res_dir / "train_config.json", "w"), indent=2)
    log(f"Done. Best val Dice {best_dice:.4f}. Params {n_params:,}.")


if __name__ == "__main__":
    main()
