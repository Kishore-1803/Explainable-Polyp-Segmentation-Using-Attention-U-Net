import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Any

# Custom modules
from src.data.split import load_split
from src.data.dataset import KvasirSegmentationDataset
from src.preprocessing.transform import JointTransform
from src.models.unet import UNet
from src.evaluation.evaluator import Evaluator, save_metrics_to_json

def main() -> None:
    # 1. Load config
    config_path = os.path.join("experiments", "unet_dropout_30", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # 2. Load dataset split
    split_path = os.path.join("experiments", "review_1", "split_indices.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Baseline split not found at {split_path}")
        
    print(f"Loading existing split from {split_path}")
    split_dict = load_split(split_path)
    
    # 3. Prepare test Dataset and DataLoader
    image_size = tuple(config["image_size"])
    test_transform = JointTransform(image_size=image_size, is_train=False)
    
    test_dataset = KvasirSegmentationDataset(
        dataset_path=config["dataset_path"],
        filenames=split_dict["test"],
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, # Evaluate image by image
        shuffle=False, 
        num_workers=0
    )
    
    # 4. Initialize Model and Load Checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")
    
    model = UNet(in_channels=3, out_channels=1, dropout_rate=config.get("dropout_rate", 0.0)).to(device)
    
    checkpoint_path = os.path.join(config["checkpoint_dir"], "review_1_best.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with Val Dice: {checkpoint['val_dice']:.4f}")
    
    # 5. Evaluation loop
    evaluator = Evaluator(model=model, test_loader=test_loader, device=device)
    metrics_summary = evaluator.evaluate(threshold=0.5)
    
    # 6. Calculate and save metrics
    results_dir = config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    save_metrics_to_json(metrics_summary, metrics_path)

if __name__ == "__main__":
    main()
