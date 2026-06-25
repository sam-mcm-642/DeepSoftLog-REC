import json

# Load your dictionary file
dict_file = '/Users/sammcmanagan/Desktop/Thesis/VG/VG-SGG-dicts.json'
with open(dict_file, 'r') as f:
    data = json.load(f)

# Convert it to the expected format
converted_data = {
    'label_to_idx': {v: int(k) for k, v in data['idx_to_label'].items()},
    'predicate_to_idx': {v: int(k) for k, v in data['idx_to_predicate'].items()},
}

# Add attributes if they exist
if 'idx_to_attribute' in data:
    converted_data['attribute_to_idx'] = {v: int(k) for k, v in data['idx_to_attribute'].items()}
else:
    converted_data['attribute_to_idx'] = {'__background__': 0}

# Ensure background classes exist with index 0
if '__background__' not in converted_data['label_to_idx']:
    converted_data['label_to_idx']['__background__'] = 0

if '__background__' not in converted_data['predicate_to_idx']:
    converted_data['predicate_to_idx']['__background__'] = 0

# Save the converted file
converted_file = '/Users/sammcmanagan/Desktop/Thesis/VG/VG-SGG-dicts-converted.json'
with open(converted_file, 'w') as f:
    json.dump(converted_data, f, indent=2)

print(f"Converted file saved to {converted_file}")
