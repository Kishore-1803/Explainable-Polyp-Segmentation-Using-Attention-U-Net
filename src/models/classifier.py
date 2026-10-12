import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Classifier(nn.Module):
    """
    ResNet50 based binary classifier for Endoscopic Lesion Detection (Normal vs. Polyp).
    Uses pre-trained ImageNet weights and replaces the final fully connected layer.
    """
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super(ResNet50Classifier, self).__init__()
        
        # Load the base model
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet50(weights=None)
            
        # Extract in_features of the original fc layer
        in_features = self.backbone.fc.in_features
        
        # Replace the fully connected layer for binary classification
        # Using a single output neuron for BCEWithLogitsLoss
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """
        Returns the last convolutional layer for Grad-CAM.
        For ResNet50, this is the last block of layer4.
        """
        return self.backbone.layer4[-1]
