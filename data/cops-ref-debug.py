import json
from collections import Counter

# Quick debug - update the path to your file
json_path = "data/data.json"  # Update this path

with open(json_path, 'r') as f:
    data = json.load(f)

if isinstance(data, dict) and 'refs' in data:
    refs = data['refs']
else:
    refs = data

print(f"Total entries: {len(refs)}")

# Count None entries
none_count = sum(1 for ref in refs if ref is None)
print(f"None entries: {none_count}")

# Check split values in first 10 valid entries
valid_refs = [ref for ref in refs if ref is not None][:10]
print(f"\nFirst 10 valid entries:")
for i, ref in enumerate(valid_refs):
    split_val = ref.get('split') if isinstance(ref, dict) else 'NOT_DICT'
    print(f"  Entry {i}: split = {split_val} (type: {type(split_val)})")

# Count all split values
all_refs = [ref for ref in refs if ref is not None and isinstance(ref, dict)]
split_values = [ref.get('split') for ref in all_refs]
split_counter = Counter(split_values)

print(f"\nSplit value counts:")
for value, count in split_counter.most_common():
    print(f"  {value}: {count:,}")