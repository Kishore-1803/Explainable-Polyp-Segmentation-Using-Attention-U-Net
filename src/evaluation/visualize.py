import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
import torch

def unnormalize_image(
    img_tensor: Union[torch.Tensor, np.ndarray], 
    mean: List[float] = [0.485, 0.456, 0.406], 
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Converts a normalized PyTorch tensor back to a standard RGB numpy image in [0, 1].
    """
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.cpu().numpy()
    else:
        img = img_tensor.copy()
        
    # Shape check: if (C, H, W) -> transpose to (H, W, C)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
        
    mean = np.array(mean)
    std = np.array(std)
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    return img

def create_visualization_grid(
    image_tensor: Union[torch.Tensor, np.ndarray],
    gt_mask: Union[torch.Tensor, np.ndarray],
    pred_prob: Union[torch.Tensor, np.ndarray],
    grad_cam_heatmap: np.ndarray,
    save_path: str,
    decoder_cam_heatmap: np.ndarray = None
) -> None:
    """
    Generates and saves a visualization figure:
    1. Original Image
    2. Ground Truth Mask
    3. Predicted Mask (thresholded at 0.5)
    4. Boundary Overlay (predicted contour in green on original image)
    5. Grad-CAM Activation Heatmap Overlay (Bottleneck)
    6. (Optional) Grad-CAM Activation Heatmap Overlay (Decoder Stage)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Prepare original image
    img_rgb = unnormalize_image(image_tensor)
    img_uint8 = (img_rgb * 255).astype(np.uint8)
    
    # 2. Prepare ground truth mask
    if isinstance(gt_mask, torch.Tensor):
        gt = gt_mask.cpu().numpy()
    else:
        gt = gt_mask.copy()
    if len(gt.shape) == 3:
        gt = gt[0]  # Remove channel dimension (H, W)
        
    # 3. Prepare predicted probability and binary mask
    if isinstance(pred_prob, torch.Tensor):
        pred = pred_prob.cpu().numpy()
    else:
        pred = pred_prob.copy()
    if len(pred.shape) == 3:
        pred = pred[0]
    pred_bin = (pred > 0.5).astype(np.uint8)
    
    # 4. Create Boundary Overlay (green contours of the prediction)
    boundary_img = img_uint8.copy()
    contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(boundary_img, contours, -1, (0, 255, 0), 2)
    boundary_img_rgb = boundary_img.astype(np.float32) / 255.0
    
    # 5. Create Grad-CAM Heatmap Overlay (Bottleneck)
    heatmap_uint8 = (grad_cam_heatmap * 255).astype(np.uint8)
    # Apply JET colormap (returns BGR on OpenCV)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend image and heatmap (60% image, 40% heatmap)
    grad_cam_overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_color_rgb, 0.4, 0)
    grad_cam_overlay_rgb = grad_cam_overlay.astype(np.float32) / 255.0
    
    # 6. Create Decoder Grad-CAM Heatmap Overlay (if provided)
    if decoder_cam_heatmap is not None:
        dec_heatmap_uint8 = (decoder_cam_heatmap * 255).astype(np.uint8)
        dec_heatmap_color = cv2.applyColorMap(dec_heatmap_uint8, cv2.COLORMAP_JET)
        dec_heatmap_color_rgb = cv2.cvtColor(dec_heatmap_color, cv2.COLOR_BGR2RGB)
        dec_grad_cam_overlay = cv2.addWeighted(img_uint8, 0.6, dec_heatmap_color_rgb, 0.4, 0)
        dec_grad_cam_overlay_rgb = dec_grad_cam_overlay.astype(np.float32) / 255.0
    
    # Plot using matplotlib
    num_cols = 6 if decoder_cam_heatmap is not None else 5
    fig, axes = plt.subplots(1, num_cols, figsize=(3.6 * num_cols, 4))
    
    # Col 1: Original Image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')
    
    # Col 2: Ground Truth Mask
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title("Ground Truth Mask", fontsize=12)
    axes[1].axis('off')
    
    # Col 3: Predicted Mask
    axes[2].imshow(pred_bin, cmap='gray')
    axes[2].set_title("Predicted Mask (0.5)", fontsize=12)
    axes[2].axis('off')
    
    # Col 4: Boundary Overlay
    axes[3].imshow(boundary_img_rgb)
    axes[3].set_title("Boundary Overlay", fontsize=12)
    axes[3].axis('off')
    
    # Col 5: Grad-CAM explanation (Bottleneck)
    axes[4].imshow(grad_cam_overlay_rgb)
    axes[4].set_title("Grad-CAM (Bottleneck)", fontsize=12)
    axes[4].axis('off')
    
    if decoder_cam_heatmap is not None:
        axes[5].imshow(dec_grad_cam_overlay_rgb)
        axes[5].set_title("Grad-CAM (Decoder)", fontsize=12)
        axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
