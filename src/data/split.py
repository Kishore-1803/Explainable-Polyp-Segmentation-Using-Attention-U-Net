import os
import json
import random
from typing import Tuple, List, Dict

def get_reproducible_split(
    dataset_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Splits the Kvasir-SEG dataset into train, validation, and test sets.
    The split is saved to prevent data leakage and ensure reproducibility.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Split ratios must sum to 1.0"
    
    images_dir = os.path.join(dataset_path, "images")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at {images_dir}")
        
    # Get all image filenames and sort them to have a consistent starting point
    filenames = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    num_samples = len(filenames)
    
    # Set seed and shuffle
    rng = random.Random(seed)
    shuffled_filenames = list(filenames)
    rng.shuffle(shuffled_filenames)
    
    # Calculate slice indices
    train_end = int(train_ratio * num_samples)
    val_end = train_end + int(val_ratio * num_samples)
    
    train_files = shuffled_filenames[:train_end]
    val_files = shuffled_filenames[train_end:val_end]
    test_files = shuffled_filenames[val_end:]
    
    split_dict = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    print(f"Dataset Split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")
    return split_dict

def save_split(split_dict: Dict[str, List[str]], save_path: str) -> None:
    """Saves the split dictionary to a JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(split_dict, f, indent=4)
    print(f"Saved split indices to {save_path}")

def load_split(split_path: str) -> Dict[str, List[str]]:
    """Loads the split dictionary from a JSON file."""
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found at {split_path}")
    with open(split_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    # Test script standalone
    config_path = os.path.join("experiments", "review_1", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    split = get_reproducible_split(
        dataset_path=config["dataset_path"],
        train_ratio=config["train_split"],
        val_ratio=config["val_split"],
        test_ratio=config["test_split"],
        seed=config["seed"]
    )
    
    save_path = os.path.join("experiments", "review_1", "split_indices.json")
    save_split(split, save_path)
