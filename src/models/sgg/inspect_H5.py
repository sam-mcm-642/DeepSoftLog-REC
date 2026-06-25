import h5py
import numpy as np
import sys
import json
import os

def inspect_h5_file(h5_path):
    """Complete inspection of an H5 file structure and content."""
    print(f"\n{'='*80}\nINSPECTING H5 FILE: {h5_path}\n{'='*80}")
    
    with h5py.File(h5_path, 'r') as f:
        print(f"\nTOP LEVEL KEYS: {list(f.keys())}")
        
        def print_attributes(obj):
            """Print all attributes of an object."""
            if len(obj.attrs) > 0:
                print(f"  Attributes:")
                for key, val in obj.attrs.items():
                    print(f"    {key}: {val}")
        
        def explore_item(name, obj):
            """Explore each item in the H5 file."""
            print(f"\n{'-'*80}\nPATH: {name}")
            
            if isinstance(obj, h5py.Group):
                print(f"TYPE: Group with {len(obj)} items")
                print_attributes(obj)
                
            elif isinstance(obj, h5py.Dataset):
                print(f"TYPE: Dataset with shape {obj.shape} and dtype {obj.dtype}")
                print_attributes(obj)
                
                # For small datasets, print sample data
                if len(obj.shape) == 0:
                    print(f"  Value: {obj[()]}")
                elif len(obj.shape) == 1 and obj.shape[0] < 100:
                    try:
                        if obj.dtype.kind == 'S' or obj.dtype.kind == 'O':  # String or object dtype
                            # Handle string data (decode bytes if needed)
                            sample = [item.decode('utf-8') if isinstance(item, bytes) else str(item) 
                                      for item in obj[:]]
                            print(f"  Data ({len(sample)} items): {sample}")
                        else:
                            print(f"  Data ({obj.shape[0]} items): {obj[:]}")
                    except Exception as e:
                        print(f"  Error reading data: {e}")
                elif len(obj.shape) == 1:
                    # For larger 1D arrays, show a few samples
                    try:
                        if obj.dtype.kind == 'S' or obj.dtype.kind == 'O':
                            sample_indices = [0, 1, 2, min(10, obj.shape[0]-1), min(20, obj.shape[0]-1)]
                            sample_indices = [i for i in sample_indices if i < obj.shape[0]]
                            sample = [str(obj[i].decode('utf-8') if isinstance(obj[i], bytes) else obj[i]) 
                                      for i in sample_indices]
                            print(f"  Sample data from indices {sample_indices}: {sample}")
                        else:
                            print(f"  Sample data: {obj[:5]}...")
                    except Exception as e:
                        print(f"  Error reading sample data: {e}")
                else:
                    # For multi-dimensional arrays, show shape and data type
                    print(f"  Multi-dimensional array, shape: {obj.shape}, dtype: {obj.dtype}")
                    try:
                        if obj.shape[0] < 5:
                            print(f"  First item: {obj[0]}")
                    except Exception as e:
                        print(f"  Error reading first item: {e}")
        
        # Visit all nodes in the file
        f.visititems(explore_item)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
    else:
        h5_path = input("Enter path to H5 file: ")
    
    inspect_h5_file(h5_path)