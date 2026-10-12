import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# Custom modules
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import get_reproducible_split, save_split, load_split
from src.data.dataset import KvasirSegmentationDataset
from src.models.attention_unet import AttentionUNet
from src.training.trainer import Trainer, BCEDiceLoss

# Configuration
CONFIG = {
    "seed": 42,
    "dataset_path": "Kvasir-SEG",
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "image_size": (256, 256),
    "batch_size": 8,
    "epochs": 30,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "dropout_rate": 0.0,
    "threshold": 0.5,
    "checkpoint_dir": "checkpoints/review_1_clahe",
    "results_dir": "results/review_1_clahe",
    "cvc_img_dir": "cvc_data/PNG/Original",
    "cvc_msk_dir": "cvc_data/PNG/Ground Truth",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": (8, 8)
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_clahe_rgb(pil_image: Image.Image) -> Image.Image:
    img_rgb = np.array(pil_image, dtype=np.uint8)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=CONFIG["clahe_clip_limit"], tileGridSize=CONFIG["clahe_tile_grid"])
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_eq)

class ClaheJointTransform:
    def __init__(self, image_size=(256, 256), is_train=True):
        self.image_size = image_size
        self.is_train = is_train

    def __call__(self, image: Image.Image, mask: Image.Image):
        # 0. Apply CLAHE
        image = apply_clahe_rgb(image)

        # 1. Resize
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            # 2. Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # 3. Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            # 4. Random rotation
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            # 5. Color Jitter
            if random.random() > 0.5:
                color_jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)
                image = color_jitter(image)

        # Convert to Tensor
        image_tensor = TF.to_tensor(image)
        mask_tensor = TF.to_tensor(mask)

        # Normalize
        image_tensor = TF.normalize(image_tensor, mean=CONFIG["mean"], std=CONFIG["std"])

        return image_tensor, mask_tensor

def compute_metrics(prob: np.ndarray, gt: np.ndarray, thr=0.5) -> dict:
    pred = prob > thr
    gt_b = gt > 0.5
    tp = np.sum(pred & gt_b)
    fp = np.sum(pred & ~gt_b)
    fn = np.sum(~pred & gt_b)
    tn = np.sum(~pred & ~gt_b)
    e = 1e-7
    return {
        "dice": float((2*tp+e) / (2*tp+fp+fn+e)),
        "iou": float((tp+e) / (tp+fp+fn+e)),
        "precision": float((tp+e) / (tp+fp+e)),
        "recall": float((tp+e) / (tp+fn+e)),
        "specificity": float((tn+e) / (tn+fp+e)),
    }

def summarize_metrics(metrics_list: list) -> dict:
    keys = ["dice", "iou", "precision", "recall", "specificity"]
    out = {}
    for k in keys:
        v = [m[k] for m in metrics_list]
        out[f"mean_{k}"] = float(np.mean(v))
        out[f"std_{k}"] = float(np.std(v))
    return out

def evaluate_dataset(model, dataloader, device, desc="Evaluating") -> dict:
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            probs = probs.cpu().numpy()
            masks = masks.cpu().numpy()
            
            for i in range(probs.shape[0]):
                m = compute_metrics(probs[i, 0], masks[i, 0], thr=CONFIG["threshold"])
                all_metrics.append(m)
                
    return summarize_metrics(all_metrics)

def evaluate_cvc(model, device) -> dict:
    print(f"\n[eval] Running CLAHE evaluation on CVC-ClinicDB (Zero-Shot)...")
    cvc_img_dir = Path(CONFIG["cvc_img_dir"])
    cvc_msk_dir = Path(CONFIG["cvc_msk_dir"])
    
    imgs = sorted(cvc_img_dir.glob("*.png"), key=lambda x: int(x.stem))
    msks = sorted(cvc_msk_dir.glob("*.png"), key=lambda x: int(x.stem))
    assert len(imgs) == 612 and len(msks) == 612
    
    all_metrics = []
    model.eval()
    
    with torch.no_grad():
        for img_path, msk_path in tqdm(zip(imgs, msks), total=len(imgs), desc="CVC Eval"):
            # Load and CLAHE-transform image
            img = Image.open(img_path).convert("RGB")
            img = apply_clahe_rgb(img)
            img = TF.resize(img, CONFIG["image_size"], interpolation=TF.InterpolationMode.BILINEAR)
            img_t = TF.normalize(TF.to_tensor(img), CONFIG["mean"], CONFIG["std"]).unsqueeze(0).to(device)
            
            # Load mask
            msk = Image.open(msk_path).convert("L")
            msk = TF.resize(msk, CONFIG["image_size"], interpolation=TF.InterpolationMode.NEAREST)
            gt = (np.array(msk, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
            
            prob = torch.sigmoid(model(img_t))[0, 0].cpu().numpy()
            m = compute_metrics(prob, gt, thr=CONFIG["threshold"])
            all_metrics.append(m)
            
    return summarize_metrics(all_metrics)

def main():
    print("="*65)
    print("  EXP-08: CLAHE-Trained Attention U-Net on Kvasir-SEG")
    print("="*65)
    
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    
    # Save config
    with open(os.path.join(CONFIG["results_dir"], "config.json"), 'w') as f:
        json.dump(CONFIG, f, indent=4)
        
    set_seed(CONFIG["seed"])
    
    # Use existing split (review_1 split_indices.json to be identical)
    split_path = "experiments/review_1/split_indices.json"
    print(f"Loading existing split from {split_path}")
    split_dict = load_split(split_path)
    
    train_transform = ClaheJointTransform(image_size=CONFIG["image_size"], is_train=True)
    val_transform = ClaheJointTransform(image_size=CONFIG["image_size"], is_train=False)
    test_transform = ClaheJointTransform(image_size=CONFIG["image_size"], is_train=False)
    
    train_dataset = KvasirSegmentationDataset(CONFIG["dataset_path"], filenames=split_dict["train"], transform=train_transform)
    val_dataset = KvasirSegmentationDataset(CONFIG["dataset_path"], filenames=split_dict["val"], transform=val_transform)
    test_dataset = KvasirSegmentationDataset(CONFIG["dataset_path"], filenames=split_dict["test"], transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=3, out_channels=1, dropout_rate=CONFIG["dropout_rate"]).to(device)
    
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=CONFIG["epochs"],
        checkpoint_dir=CONFIG["checkpoint_dir"],
        scheduler=scheduler
    )
    
    # print("\n[train] Starting training...")
    # history = trainer.fit()
    
    # Save training history
    # history_path = os.path.join(CONFIG["results_dir"], "training_history.json")
    # with open(history_path, 'w') as f:
    #     json.dump(history, f, indent=4)
        
    # Load best checkpoint for evaluation
    best_ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "review_1_clahe_best.pth")
    # Actually trainer saves it as {model_name}_best.pth. We should just find the best.pth in that dir.
    import glob
    ckpts = glob.glob(os.path.join(CONFIG["checkpoint_dir"], "*_best.pth"))
    if ckpts:
        best_ckpt_path = ckpts[0]
        print(f"\n[eval] Loading best checkpoint from {best_ckpt_path}")
        state = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state.get('model_state_dict', state))
    
    # Evaluate on Kvasir Test Set
    print("\n[eval] Evaluating on Kvasir Test Set...")
    kvasir_metrics = evaluate_dataset(model, test_loader, device, desc="Kvasir Eval")
    
    # Evaluate on CVC-ClinicDB (Zero-Shot)
    cvc_metrics = evaluate_cvc(model, device)
    
    # Save final results
    final_results = {
        "kvasir_test": kvasir_metrics,
        "cvc_test": cvc_metrics
    }
    results_path = os.path.join(CONFIG["results_dir"], "metrics.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print("\n" + "="*65)
    print("  FINAL RESULTS: EXP-08 (CLAHE-Trained)")
    print("="*65)
    print("Kvasir-SEG (In-Domain):")
    for k in ["dice", "iou", "precision", "recall", "specificity"]:
        print(f"  {k.capitalize():<12}: {kvasir_metrics[f'mean_{k}']:.4f} +/- {kvasir_metrics[f'std_{k}']:.4f}")
    
    print("\nCVC-ClinicDB (Zero-Shot):")
    for k in ["dice", "iou", "precision", "recall", "specificity"]:
        print(f"  {k.capitalize():<12}: {cvc_metrics[f'mean_{k}']:.4f} +/- {cvc_metrics[f'std_{k}']:.4f}")
    print("="*65)

if __name__ == "__main__":
    main()
