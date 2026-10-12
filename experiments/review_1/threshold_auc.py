import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from src.models.attention_unet import AttentionUNet
from src.data.dataset import KvasirSegmentationDataset
from src.preprocessing.transform import JointTransform
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_metrics(pred_prob: np.ndarray, gt: np.ndarray, thr: float) -> dict:
    pred = pred_prob > thr
    gt_b = gt > 0.5
    tp = np.sum(pred & gt_b)
    fp = np.sum(pred & ~gt_b)
    fn = np.sum(~pred & gt_b)
    tn = np.sum(~pred & ~gt_b)
    e = 1e-7
    return {
        "dice": float((2 * tp + e) / (2 * tp + fp + fn + e)),
        "iou": float((tp + e) / (tp + fp + fn + e)),
        "precision": float((tp + e) / (tp + fp + e)),
        "recall": float((tp + e) / (tp + fn + e)),
        "specificity": float((tn + e) / (tn + fp + e)),
    }

def main():
    config_path = os.path.join("experiments", "review_1", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    split_path = os.path.join("experiments", "review_1", "split_indices.json")
    if not os.path.exists(split_path):
        print("Split indices not found. Please train first.")
        return
    with open(split_path, 'r') as f:
        split_dict = json.load(f)
        
    test_files = split_dict["test"]
    image_size = tuple(config["image_size"])
    val_transform = JointTransform(image_size=image_size, is_train=False)
    
    test_dataset = KvasirSegmentationDataset(
        dataset_path=config["dataset_path"],
        filenames=test_files,
        transform=val_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = AttentionUNet(in_channels=3, out_channels=1).to(device)
    checkpoint_path = os.path.join(config["checkpoint_dir"], "review_1_best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Exiting.")
        return
        
    model.eval()
    
    all_probs = []
    all_gts = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze(0).squeeze(0)
            gts = masks.cpu().numpy().squeeze(0).squeeze(0)
            
            all_probs.append(probs)
            all_gts.append(gts)
            
    all_probs = np.array(all_probs)
    all_gts = np.array(all_gts)
    
    # Pixel-level ROC curve
    print("Computing AUC-ROC...")
    fpr, tpr, thresholds_roc = roc_curve(all_gts.flatten() > 0.5, all_probs.flatten())
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Pixel-level Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(config["results_dir"], "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")
    
    # Threshold sweep
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dice_scores = []
    
    print("Sweeping thresholds...")
    for thr in thresholds:
        dices = [compute_metrics(p, g, thr)["dice"] for p, g in zip(all_probs, all_gts)]
        dice_scores.append(np.mean(dices))
        
    plt.figure()
    plt.plot(thresholds, dice_scores, marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Mean Dice Score')
    plt.title('Dice Score vs Threshold')
    plt.grid(True)
    
    thr_path = os.path.join(config["results_dir"], "threshold_sweep.png")
    plt.savefig(thr_path)
    plt.close()
    print(f"Saved threshold sweep plot to {thr_path}")

if __name__ == "__main__":
    main()
