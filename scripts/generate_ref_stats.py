#!/usr/bin/env python3
"""
Generate reference statistics for drift monitoring.

This script computes mean and std of img_stats from training/val images.
For now, uses reasonable defaults based on typical OCT image distributions.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.monitoring.drift import img_stats
from PIL import Image


def generate_ref_stats(output_path: str = "llm/monitoring/ref_stats.npz", 
                       sample_size: int = 100):
    """
    Generate reference statistics.
    
    Args:
        output_path: Where to save ref_stats.npz
        sample_size: Number of synthetic images to sample (default: 100)
    """
    print(f"Generating reference statistics from {sample_size} synthetic samples...")
    
    # If you have real training data, replace this with actual image loading
    # For now, we'll use reasonable defaults from typical OCT images
    stats_list = []
    
    for i in range(sample_size):
        # Generate synthetic grayscale "OCT-like" image
        # Typical OCT: slightly darker, moderate std, some structure
        arr = np.random.normal(loc=0.45, scale=0.15, size=(224, 224))
        arr = np.clip(arr, 0, 1)
        
        # Simulate typical OCT sharpness (Laplacian variance ~0.02-0.08)
        noise = np.random.normal(0, 0.01, size=(224, 224))
        arr = arr + noise
        arr = np.clip(arr, 0, 1)
        
        img = Image.fromarray((arr * 255).astype(np.uint8))
        x = img_stats(img)  # [mean, std, sharp]
        stats_list.append(x)
    
    stats_array = np.array(stats_list, dtype=np.float32)  # Shape: (sample_size, 3)
    
    # Compute statistics
    mu = stats_array.mean(axis=0)      # [3]
    sigma = stats_array.std(axis=0)    # [3]
    
    print(f"\nComputed Statistics:")
    print(f"  μ (mean):  {mu}")
    print(f"  σ (std):   {sigma}")
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(output_file, mu=mu, sigma=sigma)
    print(f"\n✓ Saved to: {output_path}")


if __name__ == "__main__":
    generate_ref_stats()
