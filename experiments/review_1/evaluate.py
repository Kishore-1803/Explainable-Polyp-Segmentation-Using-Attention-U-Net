import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np

# Custom modules
from src.data.dataset import KvasirSegmentationDataset
from src.preprocessing.transform import JointTransform
from src.models.attention_unet import AttentionUNet
from src.evaluation.evaluator import Evaluator, save_metrics_to_json
from src.explainability.grad_cam import SegmentationGradCAM
from src.evaluation.visualize import create_visualization_grid

def main() -> None:
    # 1. Load config
    config_path = os.path.join("experiments", "review_1", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # 2. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Load split indices
    split_path = os.path.join("experiments", "review_1", "split_indices.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split indices not found at {split_path}. Please train the model first.")
    with open(split_path, 'r') as f:
        split_dict = json.load(f)
        
    test_files = split_dict["test"]
    print(f"Loaded {len(test_files)} test filenames.")
    
    # 4. Prepare dataset and dataloader
    image_size = tuple(config["image_size"])
    val_transform = JointTransform(image_size=image_size, is_train=False)
    
    test_dataset = KvasirSegmentationDataset(
        dataset_path=config["dataset_path"],
        filenames=test_files,
        transform=val_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Evaluate batch-by-batch with size 1 for visualization and Grad-CAM ease
        shuffle=False,
        num_workers=0
    )
    
    # 5. Instantiate model and load checkpoint
    model = AttentionUNet(in_channels=3, out_channels=1).to(device)
    checkpoint_path = os.path.join(config["checkpoint_dir"], "review_1_best.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Please run train.py first.")
        
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model weights from epoch {checkpoint['epoch']+1} with validation Dice score {checkpoint['val_dice']:.4f}")
    
    # 6. Run metrics evaluation
    evaluator = Evaluator(model=model, test_loader=test_loader, device=device)
    results = evaluator.evaluate(threshold=0.5)
    
    # Save metrics JSON
    metrics_path = os.path.join(config["results_dir"], "metrics.json")
    save_metrics_to_json(results, metrics_path)
    
    # 7. Generate Grad-CAM visualizations for the first 10 test images
    print("\nGenerating Grad-CAM visualizations for selected test images...")
    
    # We will hook into the bottleneck module and a decoder module of our Attention U-Net
    target_layer_bottleneck = model.bottleneck
    grad_cam_bottleneck = SegmentationGradCAM(model=model, target_layer=target_layer_bottleneck)
    
    # Use dec2 (128x128) as the decoder stage for a higher-resolution CAM
    target_layer_decoder = model.dec2
    grad_cam_decoder = SegmentationGradCAM(model=model, target_layer=target_layer_decoder)
    
    viz_dir = os.path.join(config["results_dir"], "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Reset model to eval mode (Grad-CAM needs eval mode, but requires grads)
    model.eval()
    
    # Run loop over the first 10 test images
    num_to_viz = min(10, len(test_dataset))
    for idx in range(num_to_viz):
        # Retrieve single item
        image_tensor, mask_tensor = test_dataset[idx]
        fname = test_files[idx]
        
        # Add batch dimension: (3, H, W) -> (1, 3, H, W)
        input_tensor = image_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        # Generate Grad-CAM heatmaps and logits prediction
        heatmaps_btnk, logits = grad_cam_bottleneck.generate_heatmap(input_tensor)
        heatmap_btnk = heatmaps_btnk[0]
        
        heatmaps_dec, _ = grad_cam_decoder.generate_heatmap(input_tensor)
        heatmap_dec = heatmaps_dec[0]
        
        # Sigmoid to get probability mask
        pred_prob = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        
        # Plot and save
        save_name = f"viz_{idx:02d}_{os.path.splitext(fname)[0]}.png"
        save_path = os.path.join(viz_dir, save_name)
        
        create_visualization_grid(
            image_tensor=image_tensor,
            gt_mask=mask_tensor,
            pred_prob=pred_prob,
            grad_cam_heatmap=heatmap_btnk,
            save_path=save_path,
            decoder_cam_heatmap=heatmap_dec
        )
        print(f"Saved visualization {idx+1}/{num_to_viz}: {save_path}")
        
    # Clean up hooks
    grad_cam_bottleneck.remove_hooks()
    grad_cam_decoder.remove_hooks()
    print("Grad-CAM evaluation and visualization task finished.")

if __name__ == "__main__":
    main()
