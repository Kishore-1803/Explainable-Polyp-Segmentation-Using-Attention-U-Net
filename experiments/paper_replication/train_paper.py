"""
Trains the full-width (paper-capacity) Attention U-Net on one preprocessing arm.

    python experiments/paper_replication/train_paper.py --arm paper
    python experiments/paper_replication/train_paper.py --arm ours

Training protocol follows the base paper where it is specified:
    Adam (lr 1e-4, b1 0.9, b2 0.999), weight decay 1e-5
    batch size 16
    cosine annealing  eta_max 1e-4 -> eta_min 1e-6
    early stopping on validation loss, patience 10
    BCE + Dice compound loss
    augmentation: rotation +/-30 deg, h/v flip, zoom 0.8-1.2, brightness +/-20%

DEVIATIONS (documented):
  * Mixed precision (fp16 autocast) is enabled. Without it the 31.4M model at
    batch 16 needs 7.2 GB and thrashes on an 8 GB laptop GPU (measured:
    254 s/epoch vs 19 s/epoch). This changes speed, not the protocol.
  * The paper sets cosine T_max = 100 while training for 50 epochs. We set
    T_max = the configured epoch count so the schedule completes.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from paper_model import PaperAttentionUNet          # noqa: E402
from training.trainer import BCEDiceLoss            # noqa: E402

CACHE = ROOT / "_cache"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ----------------------------------------------------------------- dataset
class CachedKvasir(Dataset):
    """Reads cached uint8 frames; applies the paper's augmentation on the fly."""

    def __init__(self, arm: str, which: str, augment: bool, seed: int = 42):
        manifest = json.loads((CACHE / "manifest.json").read_text())
        order = manifest["kvasir_order"]
        assign = manifest["kvasir_split"]
        self.idx = [i for i, fn in enumerate(order) if assign[fn] == which]
        self.images = np.load(CACHE / f"kvasir_{arm}_images.npy", mmap_mode="r")
        self.masks = np.load(CACHE / "kvasir_masks.npy", mmap_mode="r")
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def _augment(self, img, mask):
        import cv2
        if self.rng.random() < 0.5:
            img, mask = cv2.flip(img, 1), cv2.flip(mask, 1)
        if self.rng.random() < 0.5:
            img, mask = cv2.flip(img, 0), cv2.flip(mask, 0)

        angle = float(self.rng.uniform(-30, 30))
        scale = float(self.rng.uniform(0.8, 1.2))
        if abs(angle) > 1e-3 or abs(scale - 1.0) > 1e-3:
            M = cv2.getRotationMatrix2D((128, 128), angle, scale)
            img = cv2.warpAffine(img, M, (256, 256), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = cv2.warpAffine(mask, M, (256, 256), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        factor = float(self.rng.uniform(0.8, 1.2))          # brightness +/-20%
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return img, mask

    def __getitem__(self, i):
        j = self.idx[i]
        img = np.asarray(self.images[j])
        mask = np.asarray(self.masks[j])
        if self.augment:
            img, mask = self._augment(img, mask)
        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().div_(255.0)
        x = (x - MEAN) / STD
        y = torch.from_numpy(mask.copy()).float().unsqueeze(0)
        return x, y


# ----------------------------------------------------------------- metrics
def batch_metrics(logits, masks, eps=1e-7):
    probs = (torch.sigmoid(logits) > 0.5).float()
    dims = (1, 2, 3)
    inter = (probs * masks).sum(dims)
    union = probs.sum(dims) + masks.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    iou = (inter + eps) / (union - inter + eps)
    return dice.mean().item(), iou.mean().item()


# ----------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["paper", "ours"], required=True)
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

    run = f"paper_repl_{args.arm}"
    ckpt_dir = ROOT / "checkpoints" / run
    res_dir = ROOT / "results" / run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train.log"

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    train_ds = CachedKvasir(args.arm, "train", augment=True, seed=args.seed)
    val_ds = CachedKvasir(args.arm, "val", augment=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=amp, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=amp)

    model = PaperAttentionUNet(base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    log(f"arm={args.arm} device={device} amp={amp} params={n_params:,} "
        f"base={args.base} batch={args.batch_size} epochs={args.epochs}")
    log(f"train={len(train_ds)} val={len(val_ds)}")

    history = {k: [] for k in ("train_loss", "train_dice", "train_iou",
                               "val_loss", "val_dice", "val_iou", "lr")}
    best_dice, best_loss, stale = 0.0, float("inf"), 0

    for epoch in range(args.epochs):
        model.train()
        agg = np.zeros(3)
        n = 0
        for x, y in tqdm(train_ld, desc=f"[{args.arm}] epoch {epoch+1}/{args.epochs} train"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            d, i = batch_metrics(logits.float(), y)
            agg += np.array([loss.item(), d, i]) * x.size(0)
            n += x.size(0)
        tr_loss, tr_dice, tr_iou = agg / n

        model.eval()
        agg = np.zeros(3)
        n = 0
        with torch.no_grad():
            for x, y in tqdm(val_ld, desc=f"[{args.arm}] epoch {epoch+1}/{args.epochs} val"):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                d, i = batch_metrics(logits.float(), y)
                agg += np.array([loss.item(), d, i]) * x.size(0)
                n += x.size(0)
        va_loss, va_dice, va_iou = agg / n

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        for k, v in zip(history, (tr_loss, tr_dice, tr_iou, va_loss, va_dice, va_iou, lr_now)):
            history[k].append(float(v))

        log(f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Train Loss {tr_loss:.4f} Dice {tr_dice:.4f} IoU {tr_iou:.4f} || "
            f"Val Loss {va_loss:.4f} Dice {va_dice:.4f} IoU {va_iou:.4f} | lr {lr_now:.2e}")

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            log(f"  -> new best val Dice {best_dice:.4f}, checkpoint saved")

        if va_loss < best_loss - 1e-5:
            best_loss, stale = va_loss, 0
        else:
            stale += 1
            if stale >= args.patience:
                log(f"Early stopping after {epoch+1} epochs "
                    f"(val loss flat for {args.patience}).")
                break

    with open(res_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(res_dir / "train_config.json", "w") as f:
        json.dump({**vars(args), "params": n_params,
                   "best_val_dice": best_dice, "amp": amp}, f, indent=2)
    log(f"Done. Best val Dice {best_dice:.4f}. Params {n_params:,}.")


if __name__ == "__main__":
    main()
