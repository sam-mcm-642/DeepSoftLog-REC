#!/usr/bin/env python3
"""
Debug script to diagnose data loading issues.
Run this before evaluation to check if your data files are compatible.
"""

import sys
import json
from train.fixed_train import debug_dataset_creation, validate_data_files


def check_data_compatibility(scene_graph_file, queries_file):
    """
    Check compatibility between scene graph and queries files.
    """
    print("=== CHECKING DATA COMPATIBILITY ===")
    
    # Load queries
    try:
        with open(queries_file, 'r') as f:
            query_data = json.load(f)
        queries = query_data.get('queries', [])
        print(f"✅ Loaded {len(queries)} queries")
    except Exception as e:
        print(f"❌ Error loading queries: {e}")
        return False
    
    # Load scene graph
    import csv
    import ast
    from collections import defaultdict
    
    try:
        image_groups = defaultdict(list)
        bbox_data = defaultdict(set)  # Track available objects per image
        
        with open(scene_graph_file, 'r') as f:
            csv_reader = csv.reader(f, quotechar='"', escapechar='\\')
            next(csv_reader)  # Skip header
            
            for row in csv_reader:
                image_id = str(row[0])
                subject_name = row[1]
                subject_bbox = ast.literal_eval(row[2]) if row[2] != 'NULL' else None
                object_name = row[4]
                object_bbox = ast.literal_eval(row[5].strip()) if len(row) > 5 and row[5] and row[5].strip() and row[5] != 'NULL' else None
                
                if subject_bbox:
                    bbox_data[image_id].add((subject_name, tuple(subject_bbox)))
                if object_bbox:
                    bbox_data[image_id].add((object_name, tuple(object_bbox)))
        
        print(f"✅ Loaded scene graph data for {len(bbox_data)} images")
    except Exception as e:
        print(f"❌ Error loading scene graph: {e}")
        return False
    
    # Check compatibility
    total_queries = 0
    compatible_queries = 0
    incompatible_queries = []
    
    for query in queries:
        total_queries += 1
        image_id = str(query.get('image_id'))
        target_obj, target_bbox = query.get('target', [None, None])
        
        if image_id in bbox_data:
            target_key = (target_obj, tuple(target_bbox))
            if target_key in bbox_data[image_id]:
                compatible_queries += 1
            else:
                incompatible_queries.append({
                    'image_id': image_id,
                    'target': (target_obj, target_bbox),
                    'available': list(bbox_data[image_id])
                })
        else:
            incompatible_queries.append({
                'image_id': image_id,
                'target': (target_obj, target_bbox),
                'available': 'IMAGE_NOT_FOUND'
            })
    
    print(f"\n=== COMPATIBILITY RESULTS ===")
    print(f"Total queries: {total_queries}")
    print(f"Compatible queries: {compatible_queries}")
    print(f"Incompatible queries: {len(incompatible_queries)}")
    print(f"Compatibility rate: {compatible_queries/total_queries*100:.1f}%")
    print(f"incompatible queries: {incompatible_queries}")
    if incompatible_queries:
        print(f"\n=== SAMPLE INCOMPATIBLE QUERIES ===")
        for i, issue in enumerate(incompatible_queries[:5]):  # Show first 5
            print(f"{i+1}. Image {issue['image_id']}: Target {issue['target']}")
            if issue['available'] != 'IMAGE_NOT_FOUND':
                print(f"   Available objects: {len(issue['available'])}")
                # Show objects with same name
                same_name_objects = [obj for obj in issue['available'] if obj[0] == issue['target'][0]]
                if same_name_objects:
                    print(f"   Same name objects: {same_name_objects}")
            else:
                print(f"   Image not found in scene graph data")
    
    return len(incompatible_queries) == 0


def main():
    """Main debug function"""
    # Default paths - adjust these to your setup
    scene_graph_file = "data/gqa_scene_graphs_copsref_filtered.csv"
    queries_file = "data/final_queries.json"
    
    if len(sys.argv) >= 3:
        scene_graph_file = sys.argv[1]
        queries_file = sys.argv[2]
    
    print(f"Debugging data files:")
    print(f"  Scene graph: {scene_graph_file}")
    print(f"  Queries: {queries_file}")
    print()
    
    # Step 1: Validate files exist and have correct format
    if not validate_data_files(scene_graph_file, queries_file):
        print("❌ Data file validation failed!")
        return False
    
    print()
    
    # Step 2: Check compatibility
    if not check_data_compatibility(scene_graph_file, queries_file):
        print("❌ Data compatibility check failed!")
        print("\nSuggestions:")
        print("1. Make sure your generated queries use the same object names and bounding boxes as your scene graph")
        print("2. Check if bounding box formats match exactly")
        print("3. Verify that image_ids in queries exist in scene graph")
        return False
    
    print()
    
    # Step 3: Test actual dataset creation
    print("=== TESTING DATASET CREATION ===")
    dataset = debug_dataset_creation(scene_graph_file, queries_file, max_instances=5)
    
    if dataset and len(dataset) > 0:
        print("✅ Dataset creation successful!")
        return True
    else:
        print("❌ Dataset creation failed!")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All checks passed! Your data should work with the evaluation pipeline.")
    else:
        print("\n💥 Issues found. Please fix data compatibility before running evaluation.")
        sys.exit(1)