#!/usr/bin/env python3
"""
Script to randomly sample 40 items from the COPS ref dataset (1000 samples)
and save them as a new JSON file.
"""

import json
import random
import os
from pathlib import Path

def sample_cops_ref_dataset(input_path, output_path, sample_size=50, seed=2):
    """
    Randomly sample items from COPS ref dataset.
    
    Args:
        input_path (str): Path to the original dataset (1000 samples)
        output_path (str): Path where the sampled dataset will be saved
        sample_size (int): Number of samples to extract (default: 40)
        seed (int): Random seed for reproducibility (default: 42)
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the original dataset
    print(f"Loading dataset from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the refs array
    if 'refs' not in data:
        raise KeyError("Dataset does not contain 'refs' key")
    
    refs = data['refs']
    total_samples = len(refs)
    
    print(f"Found {total_samples} samples in the dataset")
    
    # Check if we have enough samples
    if total_samples < sample_size:
        raise ValueError(f"Dataset has only {total_samples} samples, cannot sample {sample_size}")
    
    # Randomly sample without replacement
    print(f"Randomly sampling {sample_size} items...")
    sampled_refs = random.sample(refs, sample_size)
    
    # Create new dataset structure
    sampled_data = {
        'refs': sampled_refs
    }
    
    # Add any other top-level keys from original data (excluding 'refs')
    for key, value in data.items():
        if key != 'refs':
            sampled_data[key] = value
    
    # Save the sampled dataset
    print(f"Saving sampled dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully created sample dataset with {sample_size} items")
    
    # Print some statistics
    print("\nSample Statistics:")
    print(f"- Original dataset size: {total_samples}")
    print(f"- Sample size: {sample_size}")
    print(f"- Sampling ratio: {sample_size/total_samples:.1%}")
    
    # Show a few sample IDs for verification
    sample_ids = [ref.get('ref_id', 'N/A') for ref in sampled_refs[:5]]
    print(f"- First 5 sample ref_ids: {sample_ids}")

def main():
    """Main function to run the sampling script."""
    
    # Define file paths
    input_path = "data/cops_ref_test_sample_1000.json"
    output_path = "data/cops_ref_test_sample_40.json"
    
    try:
        # Run the sampling
        sample_cops_ref_dataset(
            input_path=input_path,
            output_path=output_path,
            sample_size=40,
            seed=42  # Change this for different random samples
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())