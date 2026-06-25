import json
import pandas as pd
from collections import Counter
import os

def load_refcoco_image_ids(refcoco_path):
    """
    Load RefCOCO dataset and extract all unique image IDs
    
    Args:
        refcoco_path (str): Path to RefCOCO JSON file
    
    Returns:
        set: Set of unique image IDs from RefCOCO dataset
    """
    print(f"Loading RefCOCO data from {refcoco_path}...")
    
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
    
    print(f"Found {len(image_ids)} unique image IDs in RefCOCO dataset")
    return image_ids

def load_gqa_csv_image_ids(csv_path):
    """
    Load GQA CSV and extract all unique image IDs
    
    Args:
        csv_path (str): Path to GQA scene graphs CSV file
    
    Returns:
        set: Set of unique image IDs from GQA CSV
    """
    print(f"Loading GQA CSV data from {csv_path}...")
    
    # Read only the image_id column for efficiency
    df = pd.read_csv(csv_path, usecols=['image_id'])
    image_ids = set(df['image_id'].astype(str).unique())
    
    print(f"Found {len(image_ids)} unique image IDs in GQA CSV")
    return image_ids

def verify_image_id_coverage(refcoco_path, gqa_csv_path):
    """
    Verify that all RefCOCO image IDs are present in the GQA CSV
    
    Args:
        refcoco_path (str): Path to RefCOCO JSON file
        gqa_csv_path (str): Path to GQA scene graphs CSV file
    
    Returns:
        dict: Verification results
    """
    # Load image IDs from both datasets
    refcoco_ids = load_refcoco_image_ids(refcoco_path)
    gqa_ids = load_gqa_csv_image_ids(gqa_csv_path)
    
    # Find matches and mismatches
    found_ids = refcoco_ids.intersection(gqa_ids)
    missing_ids = refcoco_ids - gqa_ids
    extra_ids = gqa_ids - refcoco_ids
    
    # Calculate coverage percentage
    coverage_percentage = (len(found_ids) / len(refcoco_ids)) * 100 if refcoco_ids else 0
    
    # Prepare results
    results = {
        'total_refcoco_ids': len(refcoco_ids),
        'total_gqa_ids': len(gqa_ids),
        'found_ids': len(found_ids),
        'missing_ids': len(missing_ids),
        'coverage_percentage': coverage_percentage,
        'missing_id_list': sorted(list(missing_ids)),
        'extra_ids_in_gqa': len(extra_ids),
        'found_id_list': sorted(list(found_ids))
    }
    
    return results

def print_verification_report(results):
    """
    Print a detailed verification report
    
    Args:
        results (dict): Results from verify_image_id_coverage
    """
    print("\n" + "="*60)
    print("IMAGE ID VERIFICATION REPORT")
    print("="*60)
    
    print(f"RefCOCO Dataset:")
    print(f"  Total unique image IDs: {results['total_refcoco_ids']}")
    
    print(f"\nGQA CSV Dataset:")
    print(f"  Total unique image IDs: {results['total_gqa_ids']}")
    
    print(f"\nCoverage Analysis:")
    print(f"  Found in GQA CSV: {results['found_ids']}")
    print(f"  Missing from GQA CSV: {results['missing_ids']}")
    print(f"  Coverage percentage: {results['coverage_percentage']:.2f}%")
    
    if results['missing_ids'] > 0:
        print(f"\n❌ MISSING IMAGE IDs (first 20):")
        missing_sample = results['missing_id_list'][:20]
        for img_id in missing_sample:
            print(f"  - {img_id}")
        
        if len(results['missing_id_list']) > 20:
            print(f"  ... and {len(results['missing_id_list']) - 20} more")
    else:
        print(f"\n✅ ALL RefCOCO image IDs found in GQA CSV!")
    
    print(f"\nAdditional GQA images not in RefCOCO: {results['extra_ids_in_gqa']}")
    
    # Success/failure summary
    if results['missing_ids'] == 0:
        print(f"\n🎉 SUCCESS: All RefCOCO images are covered by the GQA dataset!")
    else:
        print(f"\n⚠️  WARNING: {results['missing_ids']} RefCOCO images are missing from GQA dataset")
        print(f"   This means {100 - results['coverage_percentage']:.2f}% of RefCOCO data cannot be used")

def save_verification_results(results, output_path):
    """
    Save verification results to a JSON file
    
    Args:
        results (dict): Results from verify_image_id_coverage
        output_path (str): Path to save results
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

def check_sample_refs(refcoco_path, gqa_csv_path, sample_size=10):
    """
    Check a sample of RefCOCO entries to see their corresponding GQA data
    
    Args:
        refcoco_path (str): Path to RefCOCO JSON file
        gqa_csv_path (str): Path to GQA scene graphs CSV file
        sample_size (int): Number of samples to check
    """
    print(f"\n" + "="*60)
    print(f"SAMPLE DATA CHECK ({sample_size} examples)")
    print("="*60)
    
    # Load RefCOCO data
    with open(refcoco_path, 'r') as f:
        refcoco_data = json.load(f)
    
    # Load GQA CSV
    gqa_df = pd.read_csv(gqa_csv_path)
    
    # Take sample of RefCOCO refs
    refs = refcoco_data.get('refs', [])
    sample_refs = refs[:sample_size]
    
    for i, ref in enumerate(sample_refs):
        image_id = str(ref.get('imageId') or ref.get('image_id'))
        object_name = ref.get('name', 'unknown')
        expression = ref.get('expression', 'No expression')
        
        print(f"\nSample {i+1}:")
        print(f"  Image ID: {image_id}")
        print(f"  Object: {object_name}")
        print(f"  Expression: {expression}")
        
        # Check if this image exists in GQA CSV
        gqa_rows = gqa_df[gqa_df['image_id'] == image_id]
        
        if len(gqa_rows) > 0:
            print(f"  ✅ Found {len(gqa_rows)} rows in GQA CSV")
            # Show a few example relationships
            sample_rels = gqa_rows.head(3)
            print(f"  Example relationships:")
            for _, row in sample_rels.iterrows():
                print(f"    - {row['subject']} {row['relationship']} {row['object']}")
        else:
            print(f"  ❌ No data found in GQA CSV")

def main():
    """
    Main function to run the verification
    """
    # Update these paths to match your file locations
    refcoco_path = "data/cops_ref_test_sample_1000.json"  # Update this path
    gqa_csv_path = "data/gqa_scene_graphs_formatted.csv"  # Path to your generated CSV
    results_path = "imgid_verification_results.json"
    
    # Check if files exist
    if not os.path.exists(refcoco_path):
        print(f"❌ RefCOCO file not found: {refcoco_path}")
        print("Please update the refcoco_path variable with the correct path")
        return
    
    if not os.path.exists(gqa_csv_path):
        print(f"❌ GQA CSV file not found: {gqa_csv_path}")
        print("Please make sure you've generated the GQA CSV first")
        return
    
    # Run verification
    results = verify_image_id_coverage(refcoco_path, gqa_csv_path)
    
    # Print report
    print_verification_report(results)
    
    # Save detailed results
    save_verification_results(results, results_path)
    
    # Check sample data
    check_sample_refs(refcoco_path, gqa_csv_path, sample_size=5)
    
    return results

if __name__ == "__main__":
    results = main()