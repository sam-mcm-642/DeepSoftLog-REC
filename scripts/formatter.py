"""
Preprocessor for VG scene graph CSV files.
Converts all strings to consistent camelCase format without underscores.
"""

import pandas as pd
import os
import re
import csv
from tqdm import tqdm


def format_string(s):
    """
    Format string to camelCase with:
    - First word lowercase
    - All subsequent words capitalized
    - No spaces or underscores
    
    Args:
        s (str): Input string to format
    
    Returns:
        str: Formatted string
    """
    if not s or not isinstance(s, str):
        return s
    
    # Special case for has_attribute and other common predicates
    special_cases = {
        "has_attribute": "hasAttribute",
        "next_to": "nextTo",
        "on_top_of": "onTopOf",
        "in_front_of": "inFrontOf",
        "to_the_right_of": "toTheRightOf",
        "to_the_left_of": "toTheLeftOf",
        "part_of": "partOf",
        "behind": "behind"  # No change needed, just for completeness
    }
    
    # Check if this is a special case (case-insensitive)
    lower_s = s.lower()
    if lower_s in special_cases:
        return special_cases[lower_s]
    
    # Convert to lowercase first
    s = s.lower()
    
    # Replace underscores with spaces temporarily
    s = s.replace('_', ' ')
    
    # Split by spaces and other separators
    words = re.split(r'[\s-]+', s)
    
    # First word lowercase, capitalize first letter of subsequent words
    formatted = words[0]
    for word in words[1:]:
        if word:  # Skip empty words
            formatted += word[0].upper() + word[1:] if len(word) > 1 else word.upper()
    
    return formatted


def preprocess_vg_scene_graph_csv(input_path, output_path, batch_size=50000):
    """
    Process a VG scene graph CSV file to apply consistent formatting.
    
    Args:
        input_path (str): Path to the input VG scene graph CSV
        output_path (str): Path to save the processed CSV
        batch_size (int): Number of rows to process at once
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # First, count total rows for the progress bar
    total_rows = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip header
        next(reader, None)
        for _ in reader:
            total_rows += 1
    
    print(f"Total rows in {input_path}: {total_rows}")
    
    # Process in batches
    reader = pd.read_csv(input_path, chunksize=batch_size)
    
    # Write header first
    first_chunk = True
    mode = 'w'  # Start with write mode to create/overwrite the file
    
    for i, chunk in enumerate(tqdm(reader, desc="Processing batches", unit="batch")):
        # Apply consistent formatting to all relevant columns
        
        # Format subject names
        chunk['subject'] = chunk['subject'].apply(format_string)
        
        # Format relationship names
        chunk['relationship'] = chunk['relationship'].apply(format_string)
        
        # Format object names where they're not NULL
        chunk['object'] = chunk['object'].apply(lambda x: format_string(x) if x != 'NULL' else x)
        
        # Write to output file
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode=mode)
            first_chunk = False
            mode = 'a'  # Switch to append mode for subsequent chunks
        else:
            # Append without writing the header again
            chunk.to_csv(output_path, index=False, mode=mode, header=False)
        
        print(f"Processed batch {i+1} ({min((i+1)*batch_size, total_rows)}/{total_rows} rows)")
    
    print(f"Preprocessing complete. Formatted data saved to {output_path}")


def preprocess_vg_ontology_csv(input_path, output_path):
    """
    Process a VG ontology CSV file to apply consistent formatting.
    
    Args:
        input_path (str): Path to the input ontology CSV
        output_path (str): Path to save the processed CSV
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Loading ontology from {input_path}...")
    try:
        ontology_df = pd.read_csv(input_path)
        
        # Apply consistent formatting to all columns
        ontology_df['subject'] = ontology_df['subject'].apply(format_string)
        ontology_df['relation'] = ontology_df['relation'].apply(format_string)
        ontology_df['object'] = ontology_df['object'].apply(format_string)
        
        # Save the formatted ontology
        ontology_df.to_csv(output_path, index=False)
        print(f"Preprocessed ontology saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing ontology file: {e}")


def verify_csv_formatting(file_path, sample_size=10):
    """
    Verify the formatting in a processed CSV file.
    
    Args:
        file_path (str): Path to the processed CSV file
        sample_size (int): Number of random rows to sample
    """
    print(f"Verifying formatting in {file_path}...")
    
    try:
        # Read a sample of rows
        df = pd.read_csv(file_path, nrows=1000)  # Read just enough to sample from
        
        if len(df) == 0:
            print("File is empty or couldn't be read.")
            return
        
        # Sample random rows
        sample_rows = df.sample(min(sample_size, len(df)))
        
        print(f"Sampled {len(sample_rows)} rows for verification:")
        
        for i, row in enumerate(sample_rows.itertuples()):
            print(f"Row {i+1}:")
            print(f"  Subject: '{row.subject}'")
            print(f"  Relationship: '{row.relationship}'")
            print(f"  Object: '{row.object}'")
            
            # Additional check for underscores
            columns_to_check = ['subject', 'relationship', 'object']
            for col in columns_to_check:
                val = getattr(row, col)
                if isinstance(val, str) and val != 'NULL' and ('_' in val or ' ' in val):
                    print(f"  WARNING: {col} '{val}' contains underscores or spaces!")
            
            print()
        
    except Exception as e:
        print(f"Error verifying file: {e}")


def generate_special_cases():
    """
    Generate a list of common relationship expressions that need special formatting.
    This is helpful for identifying patterns in your dataset.
    """
    common_relations = [
        "has_attribute",
        "next_to",
        "on_top_of",
        "in_front_of",
        "behind",
        "inside_of",
        "underneath",
        "above",
        "below",
        "to_the_right_of",
        "to_the_left_of",
        "part_of",
    ]
    
    print("Special cases formatting reference:")
    print("{")
    for rel in common_relations:
        formatted = format_string(rel)
        print(f'    "{rel}": "{formatted}",')
    print("}")


if __name__ == "__main__":
    # Test the formatter with some examples
    test_cases = [
        "coffee_table",
        "telephone_set",
        "has_attribute",
        "next_to",
        "on_top_of",
        "in_front_of",
        "Computer_Monitor",
        "TRAFFIC_LIGHT",
    ]
    
    print("Formatter test cases:")
    for test in test_cases:
        formatted = format_string(test)
        print(f"'{test}' → '{formatted}'")
    
    print("\nGenerating special cases reference:")
    generate_special_cases()
    
    print("\nTo process a VG scene graph CSV file, use:")
    print("preprocess_vg_scene_graph_csv('input.csv', 'output.csv')")
    
    print("\nTo process an ontology CSV file, use:")
    print("preprocess_vg_ontology_csv('ontology.csv', 'ontology_formatted.csv')")
