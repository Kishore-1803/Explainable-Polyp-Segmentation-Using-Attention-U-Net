import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import torchvision.transforms.functional as TF

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.attention_unet import AttentionUNet

# Configuration
CONFIG = {
    "baseline_ckpt": "checkpoints/review_1/review_1_best.pth",
    "clahe_ckpt": "checkpoints/review_1_clahe/review_1_best.pth",
    "cvc_img_dir": "cvc_data/PNG/Original",
    "cvc_msk_dir": "cvc_data/PNG/Ground Truth",
    "out_dir": "results/cvc_model_comparison",
    "image_size": (256, 256),
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": (8, 8),
    "threshold": 0.5
}

def apply_clahe_rgb(pil_image: Image.Image) -> Image.Image:
    img_rgb = np.array(pil_image, dtype=np.uint8)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=CONFIG["clahe_clip_limit"], tileGridSize=CONFIG["clahe_tile_grid"])
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_eq)

def compute_dice(prob, gt, thr=0.5):
    pred = prob > thr
    gt_b = gt > 0.5
    tp = np.sum(pred & gt_b)
    fp = np.sum(pred & ~gt_b)
    fn = np.sum(~pred & gt_b)
    return float((2*tp+1e-7) / (2*tp+fp+fn+1e-7))

def create_overlay(image_np, mask_np, color, alpha=0.5):
    """Creates a colored overlay on the image."""
    overlay = np.copy(image_np)
    for c in range(3):
        overlay[..., c] = np.where(mask_np > 0.5, 
                                   image_np[..., c] * (1 - alpha) + color[c] * alpha, 
                                   image_np[..., c])
    return overlay

def main():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    model_base = AttentionUNet(in_channels=3, out_channels=1).to(device)
    state_base = torch.load(CONFIG["baseline_ckpt"], map_location=device, weights_only=False)
    model_base.load_state_dict(state_base.get('model_state_dict', state_base))
    model_base.eval()
    
    model_clahe = AttentionUNet(in_channels=3, out_channels=1).to(device)
    
    # We must find the actual clahe checkpoint name because trainer might have named it something else
    import glob
    clahe_ckpts = glob.glob(os.path.join(os.path.dirname(CONFIG["clahe_ckpt"]), "*_best.pth"))
    if clahe_ckpts:
        clahe_ckpt_path = clahe_ckpts[0]
    else:
        clahe_ckpt_path = CONFIG["clahe_ckpt"]
        
    state_clahe = torch.load(clahe_ckpt_path, map_location=device, weights_only=False)
    model_clahe.load_state_dict(state_clahe.get('model_state_dict', state_clahe))
    model_clahe.eval()
    
    # Get all images
    img_dir = Path(CONFIG["cvc_img_dir"])
    msk_dir = Path(CONFIG["cvc_msk_dir"])
    imgs = sorted(img_dir.glob("*.png"), key=lambda x: int(x.stem))
    msks = sorted(msk_dir.glob("*.png"), key=lambda x: int(x.stem))
    
    # Manually selected representative IDs based on prior knowledge of CVC dataset
    # CVC has various lighting conditions. 
    # Let's pick 6 explicit frames that usually represent a good spectrum.
    selected_ids = ["3", "52", "121", "204", "380", "505"]
    
    selected_pairs = []
    for img_p, msk_p in zip(imgs, msks):
        if img_p.stem in selected_ids:
            selected_pairs.append((img_p, msk_p))
            
    print(f"Generating visual comparisons for {len(selected_pairs)} selected images...")
    
    with torch.no_grad():
        for img_path, msk_path in selected_pairs:
            img_id = img_path.stem
            
            # --- Load Data ---
            img_pil = Image.open(img_path).convert("RGB")
            msk_pil = Image.open(msk_path).convert("L")
            
            # Ground Truth
            gt_img = TF.resize(img_pil, CONFIG["image_size"], interpolation=TF.InterpolationMode.BILINEAR)
            gt_msk = TF.resize(msk_pil, CONFIG["image_size"], interpolation=TF.InterpolationMode.NEAREST)
            
            gt_img_np = np.array(gt_img)
            gt_msk_np = (np.array(gt_msk, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
            
            # --- Baseline Inference ---
            img_base_t = TF.normalize(TF.to_tensor(gt_img), CONFIG["mean"], CONFIG["std"]).unsqueeze(0).to(device)
            prob_base = torch.sigmoid(model_base(img_base_t))[0, 0].cpu().numpy()
            pred_base_np = (prob_base > CONFIG["threshold"]).astype(np.float32)
            dice_base = compute_dice(prob_base, gt_msk_np)
            
            # --- CLAHE Inference ---
            img_clahe = apply_clahe_rgb(img_pil)
            img_clahe_rsz = TF.resize(img_clahe, CONFIG["image_size"], interpolation=TF.InterpolationMode.BILINEAR)
            img_clahe_t = TF.normalize(TF.to_tensor(img_clahe_rsz), CONFIG["mean"], CONFIG["std"]).unsqueeze(0).to(device)
            prob_clahe = torch.sigmoid(model_clahe(img_clahe_t))[0, 0].cpu().numpy()
            pred_clahe_np = (prob_clahe > CONFIG["threshold"]).astype(np.float32)
            dice_clahe = compute_dice(prob_clahe, gt_msk_np)
            
            # --- Visualisation ---
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            fig.suptitle(f"CVC-ClinicDB Image ID: {img_id}", fontsize=16)
            
            # 1. Original Image + GT Overlay
            axes[0].imshow(create_overlay(gt_img_np, gt_msk_np, [0, 255, 0], 0.4))
            axes[0].set_title("Original + Ground Truth (Green)")
            axes[0].axis('off')
            
            # 2. Baseline Prediction
            axes[1].imshow(prob_base, cmap='gray')
            axes[1].set_title(f"Baseline Pred\nDice: {dice_base:.4f}")
            axes[1].axis('off')
            
            # 3. Baseline Overlay
            axes[2].imshow(create_overlay(gt_img_np, pred_base_np, [255, 0, 0], 0.4))
            axes[2].set_title("Baseline Overlay (Red)")
            axes[2].axis('off')
            
            # 4. CLAHE Prediction
            axes[3].imshow(prob_clahe, cmap='gray')
            axes[3].set_title(f"CLAHE-Trained Pred\nDice: {dice_clahe:.4f}")
            axes[3].axis('off')
            
            # 5. CLAHE Overlay
            # For the CLAHE overlay, show it on the CLAHE image so it's clear what the model saw
            axes[4].imshow(create_overlay(np.array(img_clahe_rsz), pred_clahe_np, [255, 255, 0], 0.4))
            axes[4].set_title("CLAHE Overlay (Yellow) on CLAHE Img")
            axes[4].axis('off')
            
            plt.tight_layout()
            out_file = os.path.join(CONFIG["out_dir"], f"comparison_id_{img_id}.png")
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved {out_file} | Baseline Dice: {dice_base:.4f} | CLAHE Dice: {dice_clahe:.4f}")
            
    print(f"\nAll comparisons saved to {CONFIG['out_dir']}")

if __name__ == "__main__":
    main()
