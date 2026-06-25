import json

def to_camel_case(text):
    """Convert text with spaces to camelCase"""
    words = text.split()
    if len(words) <= 1:
        return text
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

# Read the JSON file
with open('data/final_queries.json', 'r') as f:
    data = json.load(f)

# Track changes
changes_made = 0
changed_targets = []

# Process each query
for query in data['queries']:
    if 'target' in query and isinstance(query['target'], list) and len(query['target']) > 0:
        original_target = query['target'][0]
        
        # Check if target has spaces (multiple words)
        if ' ' in original_target:
            camel_case_target = to_camel_case(original_target)
            query['target'][0] = camel_case_target
            changes_made += 1
            changed_targets.append(f'"{original_target}" -> "{camel_case_target}"')

# Save the updated JSON file
with open('data/final_queries.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Conversion complete!")
print(f"Changes made: {changes_made}")
if changed_targets:
    print("Changed targets:")
    for change in changed_targets:
        print(f"  {change}")