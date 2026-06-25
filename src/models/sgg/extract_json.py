import h5py
import json
import os
import numpy as np

def extract_dictionaries_from_h5(h5_file_path, output_json_path):
    """
    Extract dictionaries from the VG-SGG-with-attri.h5 file and save them as JSON.
    
    Args:
        h5_file_path: Path to the H5 file (VG-SGG-with-attri.h5)
        output_json_path: Path to save the output JSON dictionary
    """
    print(f"Opening H5 file: {h5_file_path}")
    
    with h5py.File(h5_file_path, 'r') as f:
        # Print available keys to understand the file structure
        print("Available keys in H5 file:", list(f.keys()))
        
        # Initialize dictionaries
        idx_to_label = {}
        idx_to_predicate = {}
        idx_to_attribute = {}
        
        # Extract object labels (idx_to_label)
        if 'object_classes' in f:
            for idx, label in enumerate(f['object_classes']):
                # Decode if bytes, otherwise convert to string
                label_str = label.decode('utf-8') if isinstance(label, bytes) else str(label)
                idx_to_label[str(idx)] = label_str
            print(f"Extracted {len(idx_to_label)} object classes")
        else:
            print("Warning: 'object_classes' not found in H5 file")
            # Look for alternative keys
            for key in f.keys():
                if 'object' in key.lower() or 'class' in key.lower():
                    print(f"Possible alternative key for object classes: {key}")
        
        # Extract predicate labels (idx_to_predicate)
        if 'predicate_classes' in f:
            for idx, pred in enumerate(f['predicate_classes']):
                pred_str = pred.decode('utf-8') if isinstance(pred, bytes) else str(pred)
                idx_to_predicate[str(idx)] = pred_str
            print(f"Extracted {len(idx_to_predicate)} predicate classes")
        else:
            print("Warning: 'predicate_classes' not found in H5 file")
            # Look for alternative keys
            for key in f.keys():
                if 'predicate' in key.lower() or 'relation' in key.lower():
                    print(f"Possible alternative key for predicate classes: {key}")
        
        # Extract attribute labels (idx_to_attribute)
        if 'attribute_classes' in f:
            for idx, attr in enumerate(f['attribute_classes']):
                attr_str = attr.decode('utf-8') if isinstance(attr, bytes) else str(attr)
                idx_to_attribute[str(idx)] = attr_str
            print(f"Extracted {len(idx_to_attribute)} attribute classes")
        else:
            print("Warning: 'attribute_classes' not found in H5 file")
            # Look for alternative keys
            for key in f.keys():
                if 'attr' in key.lower():
                    print(f"Possible alternative key for attribute classes: {key}")
        
        # If the dictionaries are empty, try exploring the H5 structure further
        if not idx_to_label or not idx_to_predicate:
            print("Exploring H5 structure to find class information...")
            for key in f.keys():
                print(f"Key: {key}")
                if isinstance(f[key], h5py.Group):
                    print(f"  Subkeys: {list(f[key].keys())}")
                else:
                    print(f"  Shape: {f[key].shape}")
                    # Print sample data for small datasets
                    if len(f[key].shape) == 1 and f[key].shape[0] < 100:
                        sample = [item.decode('utf-8') if isinstance(item, bytes) else str(item) 
                                 for item in f[key][:10]]
                        print(f"  Sample data: {sample}")
        
        # Ensure '__background__' is at index 0
        if '0' not in idx_to_label or idx_to_label['0'] != '__background__':
            print("Adding __background__ to idx_to_label at index 0")
            # Shift all indices up by 1 if needed
            if idx_to_label and '0' in idx_to_label:
                shifted_idx_to_label = {}
                for idx, label in idx_to_label.items():
                    shifted_idx_to_label[str(int(idx) + 1)] = label
                idx_to_label = shifted_idx_to_label
            idx_to_label['0'] = '__background__'
        
        if '0' not in idx_to_predicate or idx_to_predicate['0'] != '__background__':
            print("Adding __background__ to idx_to_predicate at index 0")
            # Shift all indices up by 1 if needed
            if idx_to_predicate and '0' in idx_to_predicate:
                shifted_idx_to_predicate = {}
                for idx, pred in idx_to_predicate.items():
                    shifted_idx_to_predicate[str(int(idx) + 1)] = pred
                idx_to_predicate = shifted_idx_to_predicate
            idx_to_predicate['0'] = '__background__'
        
        if '0' not in idx_to_attribute or idx_to_attribute['0'] != '__background__':
            print("Adding __background__ to idx_to_attribute at index 0")
            # Shift all indices up by 1 if needed
            if idx_to_attribute and '0' in idx_to_attribute:
                shifted_idx_to_attribute = {}
                for idx, attr in idx_to_attribute.items():
                    shifted_idx_to_attribute[str(int(idx) + 1)] = attr
                idx_to_attribute = shifted_idx_to_attribute
            idx_to_attribute['0'] = '__background__'
    
    # Create the final dictionary
    vg_dict = {
        "idx_to_label": idx_to_label,
        "idx_to_predicate": idx_to_predicate,
        "idx_to_attribute": idx_to_attribute
    }
    
    # Save the dictionary to JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(vg_dict, f, indent=2)
    
    print(f"Dictionary saved to {output_json_path}")
    print(f"Number of object classes: {len(idx_to_label)}")
    print(f"Number of predicate classes: {len(idx_to_predicate)}")
    print(f"Number of attribute classes: {len(idx_to_attribute)}")
    
    # Print a few examples
    print("\nSample object classes:")
    sample_objs = list(idx_to_label.items())[:5]
    for idx, label in sample_objs:
        print(f"  {idx}: {label}")
    
    print("\nSample predicate classes:")
    sample_preds = list(idx_to_predicate.items())[:5]
    for idx, pred in sample_preds:
        print(f"  {idx}: {pred}")
    
    return vg_dict

if __name__ == "__main__":
    # Replace these paths with your actual paths
    h5_file_path = "/Users/sammcmanagan/Desktop/Thesis/VG-SGG-with-attri.h5"
    output_json_path = "data/sg/VG-SGG-dicts-with-attri.json"
    
    extract_dictionaries_from_h5(h5_file_path, output_json_path)