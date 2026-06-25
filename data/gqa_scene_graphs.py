import json
import pandas as pd
import os
from itertools import islice

LIMIT = None

def load_gqa_scene_graphs(file_path, limit=None):
    """Load GQA scene graphs from JSON file"""
    scene_graphs = []
    
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        # Take only the first limit entries if specified
        items = list(data.items())
        if limit:
            items = items[:limit]
            
        for image_id, scene_graph in items:
            # Add image_id to the scene graph data
            scene_graph['image_id'] = image_id
            scene_graphs.append(scene_graph)
    
    return scene_graphs

def format_string(text):
    """
    Basic formatting function to convert strings to camelCase format
    This is a simplified version - you may want to use your existing formatter
    """
    if not text or text == 'NULL':
        return text
    
    # Split on spaces and special characters, then join in camelCase
    words = text.lower().replace('_', ' ').replace('-', ' ').split()
    if not words:
        return text
    
    # First word lowercase, subsequent words capitalized
    formatted = words[0] + ''.join(word.capitalize() for word in words[1:])
    return formatted

def process_gqa_scene_graphs(scene_graphs):
    """Process GQA scene graphs into CSV format matching Visual Genome structure"""
    rows = []
    
    for scene_graph in scene_graphs:
        image_id = scene_graph['image_id']
        objects = scene_graph.get('objects', {})
        
        # Create a mapping of object_id to object data
        objects_dict = {}
        for obj_id, obj_data in objects.items():
            objects_dict[obj_id] = {
                'name': obj_data.get('name'),
                'bbox': [obj_data.get('x', 0), obj_data.get('y', 0), 
                        obj_data.get('w', 0), obj_data.get('h', 0)],
                'attributes': obj_data.get('attributes', []),
                'relations': obj_data.get('relations', [])
            }
        
        # Process relationships
        for subj_id, subj_data in objects_dict.items():
            subject_name = format_string(subj_data['name']) if subj_data['name'] else None
            subject_bbox = subj_data['bbox']
            
            # Add relationship rows
            for relation in subj_data['relations']:
                obj_id = relation.get('object')
                relation_name = format_string(relation.get('name'))
                
                if obj_id in objects_dict:
                    object_name = format_string(objects_dict[obj_id]['name']) if objects_dict[obj_id]['name'] else None
                    object_bbox = objects_dict[obj_id]['bbox']
                else:
                    object_name = None
                    object_bbox = None
                
                rows.append({
                    'image_id': image_id,
                    'subject': subject_name,
                    'subject_bbox': str(subject_bbox),  # Convert to string like VG format
                    'relationship': relation_name,
                    'object': object_name,
                    'object_bbox': str(object_bbox) if object_bbox else None
                })
            
            # Add attribute rows (using hasAttribute relationship like VG)
            for attr in subj_data['attributes']:
                rows.append({
                    'image_id': image_id,
                    'subject': subject_name,
                    'subject_bbox': str(subject_bbox),
                    'relationship': 'hasAttribute',
                    'object': format_string(attr),
                    'object_bbox': None  # No bounding box for attributes
                })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    return df

def main():
    """Main function to process GQA scene graphs"""
    # Update this path to your GQA scene graphs JSON file
    file_path = "/Users/sammcmanagan/Downloads/sceneGraphs/val_sceneGraphs.json"
    
    # Load GQA scene graphs
    print(f"Loading GQA scene graphs from {file_path}...")
    scene_graphs = load_gqa_scene_graphs(file_path, limit=LIMIT)
    print(f"Loaded {len(scene_graphs)} scene graphs")
    
    # Process into CSV format
    print("Processing scene graphs...")
    df = process_gqa_scene_graphs(scene_graphs)
    
    # Save to CSV
    output_path = "data/gqa_scene_graphs_formatted.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Processed {len(df)} rows and saved to {output_path}")
    
    # Display sample of the results
    print("\nSample of processed data:")
    print(df.head(10))
    
    return df

if __name__ == "__main__":
    df = main()