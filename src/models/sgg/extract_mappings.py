import h5py
import json
import os
print("TEST")
# Path to your h5 file
h5_path = '/Users/sammcmanagan/Desktop/Thesis/VG-SGG-with-attri.h5'

# Create output directory if it doesn't exist
output_dir = '/Users/sammcmanagan/Desktop/Thesis/VG'
os.makedirs(output_dir, exist_ok=True)

# Open the h5 file
with h5py.File(h5_path, 'r') as f:
    # Print the keys to see what's available
    print("Available keys in the h5 file:", list(f.keys()))
    
    # Look for keys related to class mappings
    # Common names include 'object_classes', 'predicate_classes', etc.
    
    # Example extraction (adjust based on actual keys)
    if 'object_classes' in f:
        object_classes = [x.decode('utf-8') for x in f['object_classes'][:]]
        print(f"Found {len(object_classes)} object classes")
    
    if 'predicate_classes' in f:
        predicate_classes = [x.decode('utf-8') for x in f['predicate_classes'][:]]
        print(f"Found {len(predicate_classes)} predicate classes")
        
    # Create a dictionary in the expected format
    class_dict = {
        "idx_to_label": {str(i): cls for i, cls in enumerate(object_classes)} if 'object_classes' in f else {},
        "idx_to_predicate": {str(i): pred for i, pred in enumerate(predicate_classes)} if 'predicate_classes' in f else {},
        # Add empty attribute mappings if needed
        "idx_to_attribute": {"0": "no_attribute"}
    }
    
    # Save the extracted mappings to a json file
    with open(os.path.join(output_dir, 'VG-SGG-dicts-with-attri.json'), 'w') as json_file:
        json.dump(class_dict, json_file, indent=2)
        print(f"Saved class mappings to {os.path.join(output_dir, 'VG-SGG-dicts-with-attri.json')}")