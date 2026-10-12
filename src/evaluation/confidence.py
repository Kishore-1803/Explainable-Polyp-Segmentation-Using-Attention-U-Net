import torch
import torch.nn.functional as F
import numpy as np

class ConfidenceScorer:
    """
    Computes statistically grounded confidence metrics for model predictions.
    Can be used for both segmentation (pixel-wise) and classification (image-wise).
    """
    
    @staticmethod
    def compute_binary_confidence(logits: torch.Tensor) -> torch.Tensor:
        """
        Computes prediction reliability based on distance from the decision boundary (0 for logits).
        Transforms logits into a confidence score between 0.0 (uncertain) and 1.0 (highly confident).
        
        Args:
            logits (torch.Tensor): Raw model outputs before sigmoid.
        Returns:
            torch.Tensor: Confidence scores in [0, 1].
        """
        # Probabilities
        probs = torch.sigmoid(logits)
        
        # Confidence is max(p, 1-p). 
        # If p=0.5 (uncertain), confidence = 0.5. If p=0.9 or 0.1, confidence = 0.9.
        # We rescale this from [0.5, 1.0] to [0.0, 1.0]
        confidence = 2.0 * torch.abs(probs - 0.5)
        return confidence
    
    @staticmethod
    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        """
        Computes binary cross-entropy (predictive entropy) as an uncertainty measure.
        Higher entropy = lower confidence.
        
        Args:
            logits (torch.Tensor): Raw model outputs.
        Returns:
            torch.Tensor: Entropy values.
        """
        probs = torch.sigmoid(logits)
        # Avoid log(0)
        eps = 1e-7
        entropy = - (probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
        
        # Max entropy for binary classification is ln(2) ~ 0.693. We normalize to [0, 1]
        normalized_entropy = entropy / np.log(2)
        return normalized_entropy

    @staticmethod
    def aggregate_segmentation_confidence(confidence_map: torch.Tensor, mask_pred: torch.Tensor) -> float:
        """
        Aggregates pixel-wise confidence into a single image-level confidence score.
        Focuses mostly on the predicted lesion area to determine how confident the model is
        about the lesion it found.
        """
        if mask_pred.sum() == 0:
            # If no lesion is predicted, confidence is the average background confidence
            return confidence_map.mean().item()
            
        # Average confidence only within the predicted lesion mask
        lesion_confidence = (confidence_map * mask_pred).sum() / mask_pred.sum()
        return lesion_confidence.item()
