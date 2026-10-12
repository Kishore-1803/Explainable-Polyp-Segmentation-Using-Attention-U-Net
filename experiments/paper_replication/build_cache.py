"""
Pre-computes the deterministic preprocessing for both arms and both datasets.
Run once; training and evaluation then read uint8 arrays straight from disk,
so the per-epoch cost is augmentation only.

Usage:
    python experiments/paper_replication/build_cache.py
Outputs (under Codes/_cache/):
    kvasir_paper_images.npy  (N,256,256,3) uint8      kvasir_masks.npy (N,256,256) uint8
    kvasir_ours_images.npy
    cvc_paper_images.npy                              cvc_masks.npy
    cvc_ours_images.npy
    manifest.json   -- file order, split assignment, quality-gate flags
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                       # Codes/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from paper_preprocessing import PREPROCESSORS, quality_flags, _resize_mask  # noqa: E402
from data.split import get_reproducible_split                               # noqa: E402

KVASIR_DIR = ROOT / "Kvasir-SEG"
CVC_DIR = ROOT / "cvc_data"
CACHE_DIR = ROOT / "_cache"
SEED = 42


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return m


def kvasir_pairs():
    split = get_reproducible_split(str(KVASIR_DIR), 0.8, 0.1, 0.1, SEED)
    order, assign = [], {}
    for name in ("train", "val", "test"):
        for fn in split[name]:
            order.append(fn)
            assign[fn] = name
    pairs = [(KVASIR_DIR / "images" / fn, KVASIR_DIR / "masks" / fn) for fn in order]
    return order, assign, pairs


def cvc_pairs():
    img_dir = CVC_DIR / "PNG" / "Original"
    msk_dir = CVC_DIR / "PNG" / "Ground Truth"
    imgs = {p.stem: p for p in img_dir.glob("*.png")}
    msks = {p.stem: p for p in msk_dir.glob("*.png")}
    keys = sorted(set(imgs) & set(msks), key=lambda s: int(s) if s.isdigit() else s)
    return keys, [(imgs[k], msks[k]) for k in keys]


def build(tag, pairs, names):
    n = len(pairs)
    arrays = {arm: np.zeros((n, 256, 256, 3), np.uint8) for arm in PREPROCESSORS}
    masks = np.zeros((n, 256, 256), np.uint8)
    flags = {}

    for i, (ip, mp) in enumerate(tqdm(pairs, desc=f"cache:{tag}")):
        rgb = _read_rgb(ip)
        for arm, fn in PREPROCESSORS.items():
            arrays[arm][i] = fn(rgb)
        m = _resize_mask(_read_mask(mp))
        masks[i] = (m > 127).astype(np.uint8)
        flags[names[i]] = quality_flags(cv2.resize(rgb, (256, 256)))

    CACHE_DIR.mkdir(exist_ok=True)
    for arm, arr in arrays.items():
        np.save(CACHE_DIR / f"{tag}_{arm}_images.npy", arr)
    np.save(CACHE_DIR / f"{tag}_masks.npy", masks)
    return flags


def main():
    manifest = {"seed": SEED}

    order, assign, pairs = kvasir_pairs()
    manifest["kvasir_order"] = order
    manifest["kvasir_split"] = assign
    manifest["kvasir_quality"] = build("kvasir", pairs, order)

    keys, cpairs = cvc_pairs()
    manifest["cvc_order"] = keys
    manifest["cvc_quality"] = build("cvc", cpairs, keys)

    with open(CACHE_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    kq = manifest["kvasir_quality"].values()
    cq = manifest["cvc_quality"].values()
    print(f"\nKvasir: {len(order)} images  "
          f"blurred={sum(q['blurred'] for q in kq)}  "
          f"exposure_outlier={sum(q['exposure_outlier'] for q in kq)}")
    print(f"CVC:    {len(keys)} images  "
          f"blurred={sum(q['blurred'] for q in cq)}  "
          f"exposure_outlier={sum(q['exposure_outlier'] for q in cq)}")
    print(f"Cache written to {CACHE_DIR}")


if __name__ == "__main__":
    main()
