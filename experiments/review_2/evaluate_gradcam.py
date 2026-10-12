import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.attention_unet import AttentionUNet
from src.data.dataset import KvasirSegmentationDataset
from src.preprocessing.transform import JointTransform
from src.explainability.grad_cam import SegmentationGradCAM

def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    intersection = np.sum(pred_b & gt_b)
    union = np.sum(pred_b) + np.sum(gt_b)
    return (2.0 * intersection + 1e-7) / (union + 1e-7)

def main():
    config_path = os.path.join("experiments", "review_1", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    split_path = os.path.join("experiments", "review_1", "split_indices.json")
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Target layer for Grad-CAM in AttentionUNet
    target_layer = model.up4
    grad_cam = SegmentationGradCAM(model, target_layer)
    
    dice_scores = []
    
    print("Quantifying Grad-CAM against Ground Truth...")
    for images, masks in tqdm(test_loader):
        images = images.to(device)
        masks = masks.numpy().squeeze(0).squeeze(0)
        
        heatmaps, _ = grad_cam.generate_heatmap(images)
        heatmap = heatmaps[0]
        
        # Calculate dice between heatmap (thresholded) and GT mask
        dice = compute_dice(heatmap, masks)
        dice_scores.append(dice)
        
    grad_cam.remove_hooks()
    
    mean_dice = np.mean(dice_scores)
    median_dice = np.median(dice_scores)
    
    print(f"--- Grad-CAM vs Ground Truth Segmentation ---")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Median Dice: {median_dice:.4f}")
    
    # Save results
    results_dir = os.path.join("experiments", "review_2")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "gradcam_quantification.json"), "w") as f:
        json.dump({"mean_dice": mean_dice, "median_dice": median_dice}, f, indent=4)

if __name__ == "__main__":
    main()
