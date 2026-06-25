#!/usr/bin/env python3
"""
Simple comparison script - reads from model_comparison JSON output.
"""

import json
import glob

# Find the most recent model_comparison JSON file
json_files = glob.glob("model_comparison_*.json")
if not json_files:
    print("No model_comparison_*.json file found! Run model_comparison.py first.")
    exit()

latest_file = max(json_files)
print(f"Found {len(json_files)} model_comparison files. Using the most recent: {latest_file}")
print(f"Loading: {latest_file}\n")

# Load and display results
with open(latest_file, 'r') as f:
    data = json.load(f)

for i, result in enumerate(data['detailed_results'], 1):
    print(f"{i}. {result['expression']}")

    print(f"SONNET: {result['sonnet_output']}")
    print(f"HAIKU:  {result['haiku_output']}")
