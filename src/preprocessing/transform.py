import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Tuple
from PIL import Image

class JointTransform:
    """
    Applies joint transformations to both image and mask for segmentation.
    Ensures that spatial transformations (resizing, flipping, rotation) are applied 
    identically to both, while color augmentations are applied only to the image.
    """
    def __init__(self, image_size: Tuple[int, int] = (256, 256), is_train: bool = True) -> None:
        self.image_size = image_size
        self.is_train = is_train

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Resize both image and mask
        # Mask is resized using NEAREST interpolation to preserve sharp binary boundary labels
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            # 2. Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 3. Random vertical flip
            if torch.rand(1).item() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # 4. Random rotation (90 degree increments keep boundary alignments perfect)
            if torch.rand(1).item() > 0.5:
                angle = [90, 180, 270][torch.randint(0, 3, (1,)).item()]
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # 5. Color Jitter (only applied to the image, not the mask)
            if torch.rand(1).item() > 0.5:
                color_jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)
                image = color_jitter(image)

        # Convert both to PyTorch Tensors
        # to_tensor converts PIL Image (H x W x C) in [0, 255] to Tensor (C x H x W) in [0.0, 1.0]
        image_tensor = TF.to_tensor(image)
        mask_tensor = TF.to_tensor(mask)

        # Normalize the image with ImageNet mean and std
        image_tensor = TF.normalize(
            image_tensor, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

        return image_tensor, mask_tensor
