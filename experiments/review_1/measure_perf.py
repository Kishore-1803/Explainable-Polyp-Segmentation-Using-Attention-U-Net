import os
import json
import torch
import time
from src.models.attention_unet import AttentionUNet
from src.data.dataset import KvasirSegmentationDataset
from src.preprocessing.transform import JointTransform
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def measure_performance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Measure Parameters
    model = AttentionUNet(in_channels=3, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # 2. Measure FPS
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    # Measure time
    num_iterations = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_time_per_batch = (end_time - start_time) / num_iterations
    fps = batch_size / avg_time_per_batch
    print(f"Inference FPS (batch size {batch_size}): {fps:.2f} frames/sec")
    
    # 3. Measure Peak VRAM (only if CUDA)
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak VRAM used: {peak_vram:.2f} MB")
    else:
        print("CUDA not available, VRAM not measured.")
        
    results = {
        "Total Parameters": total_params,
        "FPS": fps,
        "VRAM_MB": peak_vram if torch.cuda.is_available() else None
    }
    
    os.makedirs("results/review_1", exist_ok=True)
    with open("results/review_1/performance_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    measure_performance()
