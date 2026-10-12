import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Tuple

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy (BCE) and Dice Loss.
    Ensures robust segmentation training by balancing pixel-wise accuracy and overlap.
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard BCE Loss
        bce_loss = self.bce(logits, targets)
        
        # Dice Loss calculation
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        eps = 1e-7
        dice_coef = (2. * intersection + eps) / (union + eps)
        dice_loss = 1.0 - dice_coef
        
        # Total combined loss
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_batch_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float]:
    """
    Calculates Dice and IoU for a single batch of predictions, per-image.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    batch_size = preds.shape[0]
    preds_flat = preds.view(batch_size, -1)
    targets_flat = targets.view(batch_size, -1)
    
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    
    eps = 1e-7
    dice = (2. * intersection + eps) / (union + eps)
    
    iou_union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    iou = (intersection + eps) / (iou_union + eps)
    
    return dice.mean().item(), iou.mean().item()


class Trainer:
    """
    Trainer class to manage training and validation loops.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int,
        checkpoint_dir: str,
        scheduler: Any = None,
        early_stopping_patience: int = 10,
        log_file: str = "train.log"
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler
        self.log_file = log_file
        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Log opening message
        self.log(f"Trainer initialized. Device: {self.device}. Training for {self.epochs} epochs.")

    def log(self, message: str) -> None:
        """Helper to print message and write to log file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        print(formatted_msg)
        
        log_path = os.path.join(self.checkpoint_dir, self.log_file)
        with open(log_path, 'a') as f:
            f.write(formatted_msg + "\n")

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            dice, iou = calculate_batch_metrics(logits, masks)
            running_loss += loss.item() * images.size(0)
            running_dice += dice * images.size(0)
            running_iou += iou * images.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), dice=dice, iou=iou)
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_dice = running_dice / len(self.train_loader.dataset)
        epoch_iou = running_iou / len(self.train_loader.dataset)
        
        return epoch_loss, epoch_dice, epoch_iou

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            
            # Metrics
            dice, iou = calculate_batch_metrics(logits, masks)
            running_loss += loss.item() * images.size(0)
            running_dice += dice * images.size(0)
            running_iou += iou * images.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), dice=dice, iou=iou)
            
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_dice = running_dice / len(self.val_loader.dataset)
        epoch_iou = running_iou / len(self.val_loader.dataset)
        
        return epoch_loss, epoch_dice, epoch_iou

    def fit(self) -> Dict[str, Any]:
        best_val_dice = 0.0
        history = {
            "train_loss": [], "train_dice": [], "train_iou": [],
            "val_loss": [], "val_dice": [], "val_iou": []
        }
        
        for epoch in range(self.epochs):
            # Train and validate
            train_loss, train_dice, train_iou = self.train_epoch(epoch)
            val_loss, val_dice, val_iou = self.validate_epoch(epoch)
            
            # Step learning rate scheduler if present
            if self.scheduler:
                # If scheduler is ReduceLROnPlateau, step using loss, else standard step
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Record history
            history["train_loss"].append(train_loss)
            history["train_dice"].append(train_dice)
            history["train_iou"].append(train_iou)
            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)
            history["val_iou"].append(val_iou)
            
            # Log results
            self.log(
                f"Epoch {epoch+1:02d}/{self.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f} || "
                f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}"
            )
            
            # Checkpoint saving
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                self.epochs_without_improvement = 0
                checkpoint_path = os.path.join(self.checkpoint_dir, "review_1_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_dice': val_dice,
                    'val_iou': val_iou,
                }, checkpoint_path)
                self.log(f"----> Save New Best Checkpoint with Val Dice: {val_dice:.4f} to {checkpoint_path}")
            else:
                self.epochs_without_improvement += 1
                self.log(f"No improvement in validation dice for {self.epochs_without_improvement} epochs.")
                if self.early_stopping_patience and self.epochs_without_improvement >= self.early_stopping_patience:
                    self.log(f"Early stopping triggered after {epoch+1} epochs!")
                    break
                
        self.log(f"Training completed. Best Validation Dice: {best_val_dice:.4f}")
        return history
