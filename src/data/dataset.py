import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Callable, Optional

class KvasirSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Kvasir-SEG image-mask pairs.
    """
    def __init__(
        self,
        dataset_path: str,
        filenames: List[str],
        transform: Optional[Callable] = None
    ) -> None:
        """
        Args:
            dataset_path (str): Path to Kvasir-SEG directory.
            filenames (List[str]): List of filenames for this dataset split.
            transform (Callable, optional): Transformations to apply to images and masks.
        """
        self.dataset_path = dataset_path
        self.filenames = filenames
        self.transform = transform
        
        self.images_dir = os.path.join(dataset_path, "images")
        self.masks_dir = os.path.join(dataset_path, "masks")
        
        # Verify all files exist
        self.paired_files = []
        for fname in self.filenames:
            img_path = os.path.join(self.images_dir, fname)
            mask_path = os.path.join(self.masks_dir, fname)
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
                
            self.paired_files.append((img_path, mask_path))
            
        print(f"Initialized Kvasir-SEG Dataset with {len(self.paired_files)} verified pairs.")

    def __len__(self) -> int:
        return len(self.paired_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.paired_files[idx]
        
        # Load image and mask
        # Using PIL to be compatible with torchvision transforms
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask
        
        if self.transform:
            # Applying transformations
            # Note: For segmentation, spatial transformations (like rotation, crop, flip)
            # must be applied identically to both the image and the mask.
            # We will handle this in transform.py.
            image, mask = self.transform(image, mask)
            
        # Ensure mask is binary (0.0 or 1.0) and has shape (1, H, W)
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0.5).float()
        
        return image, mask
