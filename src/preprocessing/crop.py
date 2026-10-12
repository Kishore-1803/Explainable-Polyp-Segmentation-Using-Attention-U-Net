import torch
from typing import Tuple

def get_bounding_box(mask: torch.Tensor, padding: int = 16) -> Tuple[int, int, int, int]:
    """
    Computes the bounding box of a segmentation mask.
    Returns (ymin, ymax, xmin, xmax).
    """
    # Assuming mask shape is (..., H, W)
    if mask.sum() == 0:
        return 0, mask.shape[-2], 0, mask.shape[-1]
    
    # Get indices where mask > 0.5 (assuming values [0, 1])
    coords = torch.nonzero(mask > 0.5)
    
    if len(coords) == 0:
        return 0, mask.shape[-2], 0, mask.shape[-1]
        
    ymin, xmin = torch.min(coords, dim=0)[0][-2:]
    ymax, xmax = torch.max(coords, dim=0)[0][-2:]
    
    H, W = mask.shape[-2:]
    ymin = max(0, ymin.item() - padding)
    ymax = min(H, ymax.item() + padding)
    xmin = max(0, xmin.item() - padding)
    xmax = min(W, xmax.item() + padding)
    
    return ymin, ymax, xmin, xmax

def apply_bounding_box_crop(image: torch.Tensor, mask: torch.Tensor, padding: int = 16) -> torch.Tensor:
    """
    Crops the image based on the bounding box of the mask.
    This fulfills the Stage 02 -> Stage 03 handoff in the blueprint.
    """
    ymin, ymax, xmin, xmax = get_bounding_box(mask, padding)
    return image[..., ymin:ymax, xmin:xmax]
