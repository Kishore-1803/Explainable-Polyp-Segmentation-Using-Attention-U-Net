import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List

def calculate_metrics(pred_prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes segmentation metrics for a single pair of prediction and ground truth.
    Args:
        pred_prob (np.ndarray): Predicted probability map (values in [0, 1]).
        target (np.ndarray): Binary ground truth mask (values 0 or 1).
        threshold (float): Threshold to binarize predicted probabilities.
    Returns:
        Dict containing Dice, IoU, Precision, Recall, and Specificity.
    """
    pred_bin = (pred_prob > threshold).astype(bool)
    target_bin = (target > 0.5).astype(bool)
    
    tp = np.sum(pred_bin & target_bin)
    fp = np.sum(pred_bin & ~target_bin)
    fn = np.sum(~pred_bin & target_bin)
    tn = np.sum(~pred_bin & ~target_bin)
    
    eps = 1e-7
    
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity)
    }

class Evaluator:
    """
    Evaluator class to assess model performance on the test set.
    """
    def __init__(self, model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> None:
        self.model = model
        self.test_loader = test_loader
        self.device = device

    @torch.no_grad()
    def evaluate(self, threshold: float = 0.5) -> Dict[str, Any]:
        self.model.eval()
        
        all_metrics: List[Dict[str, float]] = []
        
        # Iterate over test set
        for images, masks in tqdm(self.test_loader, desc="Evaluating test set"):
            images = images.to(self.device)
            logits = self.model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = masks.numpy()
            
            # Compute metrics for each image in the batch
            batch_size = images.shape[0]
            for i in range(batch_size):
                pred_prob = probs[i, 0]  # First channel (shape H x W)
                target = targets[i, 0]
                
                metrics = calculate_metrics(pred_prob, target, threshold)
                all_metrics.append(metrics)
                
        # Calculate mean and standard deviation for each metric
        metric_keys = ["dice", "iou", "precision", "recall", "specificity"]
        summary_metrics = {}
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            summary_metrics[f"mean_{key}"] = float(np.mean(values))
            summary_metrics[f"std_{key}"] = float(np.std(values))
            
        print("\n--- Test Set Evaluation Summary ---")
        for key in metric_keys:
            print(f" {key.capitalize()}: {summary_metrics[f'mean_{key}']:.4f} ± {summary_metrics[f'std_{key}']:.4f}")
        print("------------------------------------\n")
        
        return {
            "summary": summary_metrics,
            "per_image": all_metrics
        }

def save_metrics_to_json(metrics_dict: Dict[str, Any], save_path: str) -> None:
    """Saves metrics dictionary to a JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Saved test evaluation metrics to {save_path}")
