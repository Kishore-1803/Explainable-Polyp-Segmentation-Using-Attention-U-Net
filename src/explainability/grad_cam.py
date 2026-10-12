import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Any

class SegmentationGradCAM:
    """
    Grad-CAM implementation tailored for semantic segmentation models.
    Computes gradients of the sum of output logits with respect to a target layer's activations
    to highlight the regions the model focused on to make the segmentation prediction.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks for activations and gradients
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)
        
    def _save_activation(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()
        
    def _save_gradient(self, module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
        # grad_output is a tuple; grad_output[0] is the gradient w.r.t. module output
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Generates the Grad-CAM heatmap for a batch of input tensors.
        Args:
            input_tensor (torch.Tensor): Preprocessed input images tensor of shape (B, 3, H, W).
        Returns:
            Tuple containing:
                - numpy array of normalized heatmaps of shape (B, H, W)
                - output logits tensor of shape (B, 1, H, W)
        """
        # Ensure model is in eval mode and gradients are cleared
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(input_tensor)
        
        # For segmentation, backpropagate the logit sum restricted to the predicted positive region.
        # This avoids gradients being dominated by background suppression.
        mask = (logits > 0).float()
        score = (logits * mask).sum()
        
        # Fallback to max logit if no positive predictions
        if score == 0:
            score = logits.max()
            
        score.backward()
        
        # Retrieve gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            raise RuntimeError("Gradients or activations were not captured. Check hooks and target layer.")
            
        # Get shape parameters
        batch_size, channels, height, width = gradients.shape
        
        # Compute neuron importance weights: Global Average Pooling of gradients per channel
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
        
        # Weighted combination of activations
        grad_cam = (weights * activations).sum(dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        
        # Apply ReLU to only keep positive activations (features that positively contribute to segmentation)
        grad_cam = F.relu(grad_cam)
        
        # Upsample Grad-CAM heatmap to match input image resolution (256x256)
        grad_cam = F.interpolate(
            grad_cam, 
            size=input_tensor.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert to numpy and normalize per-image in the batch
        grad_cam = grad_cam.cpu().numpy()
        normalized_heatmaps = []
        
        for i in range(batch_size):
            heatmap = grad_cam[i, 0]
            heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
            if heatmap_max > heatmap_min:
                heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
            else:
                heatmap = np.zeros_like(heatmap)
            normalized_heatmaps.append(heatmap)
            
        return np.stack(normalized_heatmaps), logits.detach()
        
    def remove_hooks(self) -> None:
        """Removes the registered hooks to avoid memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()
