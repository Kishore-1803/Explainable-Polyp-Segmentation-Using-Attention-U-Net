import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Custom modules
from src.data.split import get_reproducible_split, save_split, load_split
from src.data.dataset import KvasirSegmentationDataset
from src.preprocessing.transform import JointTransform
from src.models.attention_unet import AttentionUNet
from src.training.trainer import Trainer, BCEDiceLoss

def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across CPU and GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_learning_curves(history: dict, save_path: str) -> None:
    """Plots training and validation loss, Dice, and IoU curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss Curve
    axes[0].plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    axes[0].plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Dice Curve
    axes[1].plot(epochs, history["train_dice"], 'b-', label='Train Dice')
    axes[1].plot(epochs, history["val_dice"], 'r-', label='Val Dice')
    axes[1].set_title('Training & Validation Dice Score')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Dice Score')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. IoU Curve
    axes[2].plot(epochs, history["train_iou"], 'b-', label='Train IoU')
    axes[2].plot(epochs, history["val_iou"], 'r-', label='Val IoU')
    axes[2].set_title('Training & Validation IoU')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Intersection over Union (IoU)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved learning curves to {save_path}")

def main() -> None:
    # 1. Load config
    config_path = os.path.join("experiments", "dropout_0.2", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # 2. Set seed
    set_seed(config["seed"])
    
    # 3. Handle dataset split
    split_path = os.path.join("experiments", "review_1", "split_indices.json")
    if not os.path.exists(split_path):
        print("Generating new reproducible split...")
        split_dict = get_reproducible_split(
            dataset_path=config["dataset_path"],
            train_ratio=config["train_split"],
            val_ratio=config["val_split"],
            test_ratio=config["test_split"],
            seed=config["seed"]
        )
        save_split(split_dict, split_path)
    else:
        print(f"Loading existing split from {split_path}")
        split_dict = load_split(split_path)
        
    # 4. Prepare transforms and Datasets
    image_size = tuple(config["image_size"])
    train_transform = JointTransform(image_size=image_size, is_train=True)
    val_transform = JointTransform(image_size=image_size, is_train=False)
    
    train_dataset = KvasirSegmentationDataset(
        dataset_path=config["dataset_path"],
        filenames=split_dict["train"],
        transform=train_transform
    )
    
    val_dataset = KvasirSegmentationDataset(
        dataset_path=config["dataset_path"],
        filenames=split_dict["val"],
        transform=val_transform
    )
    
    # 5. Create DataLoaders
    # Note: Adjust num_workers if needed; 0 is safest on Windows inside venv
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    # 6. Initialize Model and Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=3, out_channels=1, dropout_rate=0.2).to(device)
    
    # 7. Training configuration
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"], 
        weight_decay=config["weight_decay"]
    )
    
    # LR Scheduler (Reduce LR when validation loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # 8. Initialize Trainer and fit
    checkpoint_dir = config["checkpoint_dir"]
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"],
        checkpoint_dir=checkpoint_dir,
        scheduler=scheduler
    )
    
    history = trainer.fit()
    
    # Save training history as JSON
    history_path = os.path.join(config["results_dir"], "training_history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history JSON to {history_path}")
    
    # Plot learning curves
    learning_curves_path = os.path.join(config["results_dir"], "learning_curves.png")
    plot_learning_curves(history, learning_curves_path)

if __name__ == "__main__":
    main()
