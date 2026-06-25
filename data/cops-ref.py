import json
from collections import defaultdict, Counter

def analyze_cops_ref_splits(json_path):
    """
    Analyze and extract train/val/test splits from Cops-Ref dataset
    """
    print("Loading Cops-Ref dataset...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'refs' in data:
        refs = data['refs']
    elif isinstance(data, list):
        refs = data
    else:
        raise ValueError("Unexpected JSON structure")
    
    # Filter out null entries
    valid_refs = [ref for ref in refs if ref is not None and 'split' in ref]
    
    # Count splits
    split_counts = Counter(ref['split'] for ref in valid_refs)
    splitOld_counts = Counter(ref['splitOld'] for ref in valid_refs)
    
    print("\n" + "="*50)
    print("COPS-REF SPLIT ANALYSIS")
    print("="*50)
    
    print(f"\nTotal valid entries: {len(valid_refs)}")
    
    print(f"\nCops-Ref splits ('split' field):")
    for split_name, count in sorted(split_counts.items()):
        print(f"  {split_name}: {count:,} expressions")
    
    print(f"\nOriginal GQA splits ('splitOld' field):")
    for split_name, count in sorted(splitOld_counts.items()):
        print(f"  {split_name}: {count:,} expressions")
    
    # Cross-tabulation
    print(f"\nCross-tabulation (Cops-Ref vs Original GQA):")
    cross_tab = defaultdict(lambda: defaultdict(int))
    for ref in valid_refs:
        cross_tab[ref['split']][ref['splitOld']] += 1
    
    for cops_split in sorted(cross_tab.keys()):
        print(f"\nCops-Ref '{cops_split}':")
        for gqa_split, count in sorted(cross_tab[cops_split].items()):
            print(f"  from GQA '{gqa_split}': {count:,}")
    
    return split_counts, cross_tab

def extract_split_data(json_path, output_dir="./splits"):
    """
    Extract separate JSON files for each split
    """
    import os
    
    print(f"\nExtracting split files to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'refs' in data:
        refs = data['refs']
    elif isinstance(data, list):
        refs = data
    else:
        raise ValueError("Unexpected JSON structure")
    
    # Group by split
    splits = defaultdict(list)
    for ref in refs:
        if ref is not None and 'split' in ref:
            splits[ref['split']].append(ref)
    
    # Save each split
    for split_name, split_refs in splits.items():
        output_file = os.path.join(output_dir, f"cops_ref_{split_name}.json")
        
        # Maintain original structure if it was nested
        if isinstance(data, dict) and 'refs' in data:
            split_data = {"refs": split_refs}
        else:
            split_data = split_refs
            
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"  {split_name}: {len(split_refs):,} entries → {output_file}")
    
    return splits

def verify_paper_numbers(split_counts):
    """
    Verify if the counts match the paper's reported numbers
    """
    print(f"\n" + "="*50)
    print("VERIFICATION AGAINST PAPER")
    print("="*50)
    
    paper_numbers = {
        'train': 119603,
        'val': 16524, 
        'test': 12586
    }
    
    print("Expected (from paper) vs Actual:")
    total_expected = sum(paper_numbers.values())
    total_actual = sum(split_counts.values())
    
    for split in ['train', 'val', 'test']:
        expected = paper_numbers.get(split, 0)
        actual = split_counts.get(split, 0)
        match = "✓" if expected == actual else "✗"
        print(f"  {split}: {expected:,} vs {actual:,} {match}")
    
    print(f"\nTotal: {total_expected:,} vs {total_actual:,}")
    
    if total_expected == total_actual:
        print("✅ All numbers match the paper!")
    else:
        print("⚠️  Numbers don't match - check data completeness")

def main():
    # Update this path to your Cops-Ref JSON file
    json_path = "/Users/sammcmanagan/Downloads/data.json"
    
    try:
        # Analyze splits
        split_counts, cross_tab = analyze_cops_ref_splits(json_path)
        
        # Verify against paper
        verify_paper_numbers(split_counts)
        
        # Extract split files
        splits = extract_split_data(json_path)
        
        print(f"\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print("✅ Split analysis complete")
        print("✅ Individual split files created in ./splits/")
        print("✅ Ready for train/val/test usage")
        
        # Usage example
        print(f"\nUsage example:")
        print(f"```python")
        print(f"# Load training data")
        print(f"with open('./splits/cops_ref_train.json', 'r') as f:")
        print(f"    train_data = json.load(f)")
        print(f"")
        print(f"# Load validation data")  
        print(f"with open('./splits/cops_ref_val.json', 'r') as f:")
        print(f"    val_data = json.load(f)")
        print(f"```")
        
    except FileNotFoundError:
        print(f"❌ File not found: {json_path}")
        print("Please update the json_path variable with the correct path to your Cops-Ref data file")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()