import json
import pandas as pd
import os

def load_refcoco_image_ids(refcoco_path):
    """
    Load RefCOCO dataset and extract all unique image IDs
    
    Args:
        refcoco_path (str): Path to RefCOCO JSON file
    
    Returns:
        set: Set of unique image IDs from RefCOCO dataset
    """
    print(f"Loading RefCOCO image IDs from {refcoco_path}...")
    
    with open(refcoco_path, 'r') as f:
        refcoco_data = json.load(f)
    
    # Extract image IDs from the refs
    image_ids = set()
    refs = refcoco_data.get('refs', [])
    
    for ref in refs:
        # RefCOCO uses both 'imageId' and 'image_id' - they should be the same
        image_id = ref.get('imageId') or ref.get('image_id')
        if image_id:
            image_ids.add(str(image_id))  # Convert to string for consistency
    
    print(f"Found {len(image_ids)} unique RefCOCO image IDs")
    return image_ids

def filter_gqa_csv_by_image_ids(gqa_csv_path, refcoco_image_ids, output_path):
    """
    Filter GQA CSV to only include rows with RefCOCO image IDs
    
    Args:
        gqa_csv_path (str): Path to the full GQA CSV
        refcoco_image_ids (set): Set of RefCOCO image IDs to keep
        output_path (str): Path to save the filtered CSV
    
    Returns:
        pandas.DataFrame: Filtered dataframe
    """
    print(f"Loading GQA CSV from {gqa_csv_path}...")
    
    # Load the full GQA CSV
    df = pd.read_csv(gqa_csv_path)
    
    # Convert image_id column to string for consistent comparison
    df['image_id'] = df['image_id'].astype(str)
    
    print(f"Original GQA CSV has {len(df)} rows with {df['image_id'].nunique()} unique image IDs")
    
    # Filter to only include RefCOCO image IDs
    filtered_df = df[df['image_id'].isin(refcoco_image_ids)]
    
    print(f"Filtered GQA CSV has {len(filtered_df)} rows")
    
    # Save filtered CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    
    print(f"Filtered CSV saved to {output_path}")
    
    return filtered_df

def verify_filtered_csv(filtered_df):
    """
    Verify the filtered CSV has the expected number of unique image IDs
    
    Args:
        filtered_df (pandas.DataFrame): The filtered dataframe
    """
    unique_image_ids = filtered_df['image_id'].nunique()
    
    print(f"\nVERIFICATION:")
    print(f"Number of unique image IDs in filtered CSV: {unique_image_ids}")
    
    # Assert it's 899 as expected
    expected_count = 899
    assert unique_image_ids == expected_count, f"Expected {expected_count} unique image IDs, but found {unique_image_ids}"
    
    print(f"✅ SUCCESS: Filtered CSV contains exactly {expected_count} unique image IDs as expected!")
    
    return unique_image_ids

def main():
    """
    Main function to filter GQA CSV by RefCOCO image IDs
    """
    # File paths - update these to match your setup
    refcoco_path = "data/cops_ref_test_sample_1000.json"  # Update this path
    gqa_csv_path = "data/gqa_scene_graphs_formatted.csv"  # Your full GQA CSV
    filtered_output_path = "data/gqa_scene_graphs_copsref_filtered.csv"  # Output path
    
    # Check if input files exist
    if not os.path.exists(refcoco_path):
        print(f"❌ RefCOCO file not found: {refcoco_path}")
        print("Please update the refcoco_path variable with the correct path")
        return
    
    if not os.path.exists(gqa_csv_path):
        print(f"❌ GQA CSV file not found: {gqa_csv_path}")
        print("Please make sure you've generated the full GQA CSV first")
        return
    
    # Load RefCOCO image IDs
    refcoco_image_ids = load_refcoco_image_ids(refcoco_path)
    
    # Filter GQA CSV
    filtered_df = filter_gqa_csv_by_image_ids(gqa_csv_path, refcoco_image_ids, filtered_output_path)
    
    # Verify the result
    unique_count = verify_filtered_csv(filtered_df)
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULT: {unique_count} unique image IDs")
    print(f"{'='*50}")
    
    return filtered_df

if __name__ == "__main__":
    filtered_df = main()