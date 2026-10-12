import torch
from PIL import Image
import numpy as np

# Custom modules
from src.models.attention_unet import AttentionUNet
from src.preprocessing.transform import JointTransform

def test_model_forward_backward() -> None:
    print("--- Testing Attention U-Net Forward/Backward Pass ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize model
    model = AttentionUNet(in_channels=3, out_channels=1).to(device)
    model.train()
    
    # 2. Create dummy input and target mask
    # Batch size = 2, Channels = 3, Height = 256, Width = 256
    dummy_input = torch.randn(2, 3, 256, 256, device=device, requires_grad=True)
    dummy_target = torch.randint(0, 2, (2, 1, 256, 256), dtype=torch.float32, device=device)
    
    # 3. Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (2, 1, 256, 256), f"Expected output shape (2, 1, 256, 256), but got {output.shape}"
    print("Forward pass completed successfully.")
    
    # 4. Backward pass (sanity check)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(output, dummy_target)
    loss.backward()
    
    # Check if gradients are populated
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} does not have gradients!"
            
    print("Backward pass and gradient backpropagation completed successfully.")
    print("--- Model Verification Passed ---\n")

def test_joint_transforms() -> None:
    print("--- Testing Preprocessing & JointTransform ---")
    # 1. Create a dummy image and mask
    dummy_img = Image.fromarray(np.random.randint(0, 256, (400, 500, 3), dtype=np.uint8))
    dummy_mask = Image.fromarray(np.random.randint(0, 2, (400, 500), dtype=np.uint8) * 255)
    
    # 2. Initialize transform (train mode with augmentations)
    transform = JointTransform(image_size=(256, 256), is_train=True)
    
    # 3. Apply transform
    img_tensor, mask_tensor = transform(dummy_img, dummy_mask)
    
    print(f"Original image shape: {dummy_img.size} (W, H)")
    print(f"Transformed image shape: {img_tensor.shape} (C, H, W)")
    print(f"Transformed mask shape: {mask_tensor.shape} (C, H, W)")
    
    assert img_tensor.shape == (3, 256, 256), f"Expected image tensor shape (3, 256, 256), got {img_tensor.shape}"
    assert mask_tensor.shape == (1, 256, 256), f"Expected mask tensor shape (1, 256, 256), got {mask_tensor.shape}"
    
    # Check normalization
    assert img_tensor.min() < 0.0 or img_tensor.max() > 1.0, "Image tensor should be normalized (ImageNet mean/std)."
    # Check mask values are strictly binary 0 or 1
    unique_vals = torch.unique(mask_tensor).tolist()
    print(f"Unique values in transformed mask: {unique_vals}")
    for val in unique_vals:
        assert val in [0.0, 1.0], f"Mask tensor should only contain 0.0 or 1.0, but got value {val}"
        
    print("--- Preprocessing Verification Passed ---\n")

if __name__ == "__main__":
    try:
        test_model_forward_backward()
        test_joint_transforms()
        print("ALL PIPELINE SANITY CHECKS PASSED.")
    except Exception as e:
        print(f"Sanity Check FAILED: {str(e)}")
        import sys
        sys.exit(1)
