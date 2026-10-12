import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Any

class ClassifierGradCAM:
    """
    Grad-CAM implementation tailored for classification models (e.g., ResNet50).
    Computes gradients of the target class logit with respect to a target layer's activations
    to highlight the regions the model focused on to make the classification decision.
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
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Generates the Grad-CAM heatmap for a batch of input tensors.
        Args:
            input_tensor (torch.Tensor): Preprocessed input images tensor (B, 3, H, W).
            target_class (int, optional): The class index to visualize. If None, uses the predicted class.
                                          For binary (1 output), usually 0 for the single logit.
        Returns:
            Tuple containing:
                - numpy array of normalized heatmaps of shape (B, H, W)
                - output logits tensor
        """
        self.model.eval()
        self.model.zero_grad()
        
        logits = self.model(input_tensor) # Shape: (B, num_classes)
        
        batch_size = input_tensor.size(0)
        
        # Determine the target scores to backpropagate
        if target_class is None:
            # For binary classification with 1 output neuron, the score is just the logit
            if logits.shape[1] == 1:
                scores = logits
            else:
                scores = logits.max(dim=1, keepdim=True)[0]
        else:
            if logits.shape[1] == 1:
                # If binary and target is 1, use logit. If 0, use -logit.
                scores = logits if target_class == 1 else -logits
            else:
                scores = logits[:, target_class].unsqueeze(1)
                
        # Sum the scores over the batch for backward pass
        score_sum = scores.sum()
        score_sum.backward()
        
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            raise RuntimeError("Gradients or activations were not captured.")
            
        # Global Average Pooling of gradients per channel
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
        
        # Weighted combination of activations
        grad_cam = (weights * activations).sum(dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        
        # Apply ReLU to keep only positive contributions
        grad_cam = F.relu(grad_cam)
        
        # Upsample to match input image resolution
        grad_cam = F.interpolate(
            grad_cam, 
            size=input_tensor.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize heatmaps
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
        self.forward_hook.remove()
        self.backward_hook.remove()
