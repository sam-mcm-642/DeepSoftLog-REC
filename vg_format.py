
"""
Batch processor for VG scene graphs.
Handles large VG scene graph files by processing them in manageable chunks.
"""

import pandas as pd
import json
import os
from pathlib import Path
import csv
from tqdm import tqdm
from formatter import format_string
import random


def preprocess_vg_scene_graph_in_batches(input_path, output_path, batch_size=50000):
    """
    Process a large VG scene graph CSV file in batches to format object names and relationships.
    
    Args:
        input_path (str): Path to the input VG scene graph CSV
        output_path (str): Path to save the processed CSV
        batch_size (int): Number of rows to process at once
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # First, count total rows to provide progress information
    total_rows = 0
    with open(input_path, 'r') as f:
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
        # Format the data in this chunk
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


def verify_formatted_scene_graph(file_path, sample_size=10):
    """
    Verify the formatting of a processed scene graph file by checking a sample of rows.
    
    Args:
        file_path (str): Path to the processed scene graph file
        sample_size (int): Number of random rows to sample for verification
    """
    print(f"Verifying formatting in {file_path}...")
    
    # Check if it's CSV or JSON
    if file_path.endswith('.csv'):
        # Read a sample of rows for verification
        df = pd.read_csv(file_path, nrows=1000)  # Read just enough to sample from
        
        if len(df) == 0:
            print("File is empty or couldn't be read.")
            return
        
        # Sample random rows
        sample_rows = df.sample(min(sample_size, len(df)))
        
        print(f"Sampled {len(sample_rows)} rows for verification:")
        
        format_issues = False
        
        for _, row in sample_rows.iterrows():
            print(f"Subject: '{row['subject']}'")
            print(f"Relationship: '{row['relationship']}'")
            print(f"Object: '{row['object']}'")
            
            # Check if formatting is correct using the fixed is_properly_formatted function
            if not is_properly_formatted(row['subject']):
                format_issues = True
                print(f"Warning: Subject '{row['subject']}' is not properly formatted.")
                print(f"Should be formatted as camelCase with first word lowercase.")
            
            if not is_properly_formatted(row['relationship']):
                format_issues = True 
                print(f"Warning: Relationship '{row['relationship']}' is not properly formatted.")
                print(f"Should be formatted as camelCase with first word lowercase.")
            
            # Only check object if it's not NULL
            if row['object'] != 'NULL' and not is_properly_formatted(row['object']):
                format_issues = True
                print(f"Warning: Object '{row['object']}' is not properly formatted.")
                print(f"Should be formatted as camelCase with first word lowercase.")
            
            print("---")
        
        if not format_issues:
            print("All sampled rows are properly formatted.")
    
    elif file_path.endswith('.json'):
        # Load a sample from the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            print("File is empty or couldn't be read.")
            return
        
        # Sample random scene graphs
        sample_data = random.sample(data, min(sample_size, len(data)))
        
        print(f"Sampled {len(sample_data)} scene graphs for verification:")
        
        format_issues = False
        
        for sg in sample_data:
            if 'objects' in sg and sg['objects']:
                for obj in sg['objects'][:2]:  # Check just a couple of objects
                    if 'name' in obj:
                        print(f"Object name: '{obj['name']}'")
                        
                        # Check if formatting is correct
                        if not is_properly_formatted(obj['name']):
                            format_issues = True
                            print(f"Warning: Object name '{obj['name']}' is not properly formatted.")
                            print(f"Should be formatted as camelCase with first word lowercase.")
            
            if 'relationships' in sg and sg['relationships']:
                for rel in sg['relationships'][:2]:  # Check just a couple of relationships
                    if 'predicate' in rel:
                        print(f"Relationship: '{rel['predicate']}'")
                        
                        # Check if formatting is correct
                        if not is_properly_formatted(rel['predicate']):
                            format_issues = True
                            print(f"Warning: Relationship '{rel['predicate']}' is not properly formatted.")
                            print(f"Should be formatted as camelCase with first word lowercase.")
            
            print("---")
        
        if not format_issues:
            print("All sampled data is properly formatted.")
    
    else:
        print(f"Unsupported file format: {file_path}")


def verify_generated_queries(queries_path):
    """
    Verify that the generated queries have consistent formatting.
    
    Args:
        queries_path (str): Path to the generated queries JSON file
    """
    print(f"Verifying formatting in generated queries from {queries_path}...")
    
    with open(queries_path, 'r') as f:
        data = json.load(f)
    
    queries = data.get('queries', [])
    
    if not queries:
        print("No queries found to verify.")
        return
    
    print(f"Loaded {len(queries)} queries for verification.")
    
    # Check a sample of queries
    sample_size = min(10, len(queries))
    sample_queries = queries[:sample_size]
    
    format_issues = False
    
    for i, query_data in enumerate(sample_queries):
        print(f"Sample {i+1}:")
        print(f"  Image ID: {query_data['image_id']}")
        print(f"  Query: {query_data['query']}")
        print(f"  Target: {query_data['target'][0]}")
        
        # Extract and check type predicate format
        type_match = re.search(r'type\(X, ([^)]+)\)', query_data['query'])
        if type_match:
            type_entity = type_match.group(1)
            if not is_properly_formatted(type_entity):
                format_issues = True
                print(f"  WARNING: Type entity '{type_entity}' is not properly formatted.")
        
        # Extract and check expression predicates
        expr_matches = re.findall(r'expression\(([^,]+), [^,]+, ([^)]+)\)', query_data['query'])
        for relation, entity in expr_matches:
            if not is_properly_formatted(relation):
                format_issues = True
                print(f"  WARNING: Relation '{relation}' is not properly formatted.")
            
            # Only check entity if it's not a variable (like X)
            if entity != 'X' and not is_properly_formatted(entity):
                format_issues = True
                print(f"  WARNING: Entity '{entity}' is not properly formatted.")
        
        # Check target object formatting
        if not is_properly_formatted(query_data['target'][0]):
            format_issues = True
            print(f"  WARNING: Target object '{query_data['target'][0]}' is not properly formatted.")
        
        print()
    
    if not format_issues:
        print("All sampled queries are properly formatted.")


def main():
    """Run the batch processor on VG scene graph data."""
    # Define paths (override via config/CLI as needed)
    vg_scene_graph_path = "data/vg/vg_scene_graphs.csv"  # Original VG scene graph
    processed_scene_graph_path = "data/vg/vg_scene_graphs_formatted.csv"  # Processed scene graph
    
    # Process in batches
    preprocess_vg_scene_graph_in_batches(
        vg_scene_graph_path,
        processed_scene_graph_path,
        batch_size=50000  # Adjust based on your available memory
    )
    
    # Verify the formatting
    # verify_formatted_scene_graph(processed_scene_graph_path, sample_size=5)


if __name__ == "__main__":
    import re  # Import re for query verification
    main()