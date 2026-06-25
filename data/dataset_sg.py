from deepsoftlog.data.dataset import Dataset
from deepsoftlog.data import sg_to_prolog
from dataclasses import dataclass
from typing import List, Dict, Union, Iterable, Tuple
from collections import defaultdict
import json
import csv
import ast
import pandas as pd

@dataclass
class SceneGraph:
    """
    Represents a scene graph for an instance.
    """
    triplets: List[List[str]]  # List of triplets ['bbox1', 'on', 'bbox2']
    bounding_boxes: Dict[str, Tuple[str, List[int]]]  # {bbox_id: (object_name, [x1,y1,x2,y2])}
    attributes: Dict[str, str] = None


@dataclass
class DatasetInstance:
    """
    Represents a single instance in the dataset.
    """
    query: str  # Prolog query, e.g., 'target(X), is(X, man), (X, nextTo, woman), (woman, wearing, shirt) .'
    scene_graph: SceneGraph  # The scene graph associated with the image
    target: tuple[str, str]  # For generated scene graphs: (target_object, None)
    metadata: Dict[str, Union[str, int, float]] = None  # Now includes 'target_bbox' coordinates
    
    
class ReferringExpressionDataset(Dataset):
    """
    Custom Dataset for referring expression tasks.
    Extends the abstract Dataset class from DeepSoftLog.
    Updated to handle generated scene graphs with IoU-based matching.
    """
    def __init__(self, instances: List[DatasetInstance]):
        self.instances = list(instances)

    def __len__(self):
        """
        Returns the number of instances in the dataset.
        """
        return len(self.instances)

    def __getitem__(self, idx: int) -> DatasetInstance:
        """
        Returns a specific instance by index.
        """
        return self.instances[idx]

    def add_instance(self, instance: DatasetInstance):
        """
        Adds a new instance to the dataset.
        """
        self.instances.append(instance)

    def __str__(self):
        """
        String representation of the dataset showing the first few instances.
        """
        nb_rows = min(len(self), 5)
        return "\n".join(str(self[i]) for i in range(nb_rows))
    
    def generate_data_instances(self, filepath) -> List[DatasetInstance]:
        """
        Generate dataset instances from CSV file.
        Updated for generated scene graphs - no longer relies on exact bbox matching.
        """
        
        # Load queries
        with open("final_queries_edited.json", "r") as f_queries:
            query_data = json.load(f_queries)
            
        image_groups = defaultdict(list)
        
        # Read CSV with proper quoting to handle brackets
        with open(filepath, 'r') as f:
            csv_reader = csv.reader(f, quotechar='"', escapechar='\\')
            # Skip header
            next(csv_reader)
            
            for row in csv_reader:
                # Convert string bounding boxes to lists of integers
                subject_bbox = ast.literal_eval(row[2]) if row[2] != 'NULL' else None
                object_bbox = ast.literal_eval(row[5].strip()) if len(row) > 5 and row[5] and row[5].strip() and row[5] != 'NULL' else None
                
                processed_row = [
                    str(row[0]),  # image_id
                    row[1],       # subject
                    subject_bbox,
                    row[3],       # relationship
                    row[4],       # object
                    object_bbox
                ]
                
                image_groups[processed_row[0]].append(processed_row)
        
        # Process each image group
        for image_id, scene_rows in image_groups.items():
            bbox_dict = {}
            bbox_id_map = {}  # Maps (object_name, bbox_tuple) -> bbox_id
            attr_dict = {}    # Maps attribute_name -> att_id
            attr_id_map = {}  # Maps attribute_name -> att_id
            
            bbox_counter = 1
            attr_counter = 1  # Counter for attributes
            
            for row in scene_rows:
                _, subject_name, subject_bbox, relationship, object_name, object_bbox = row
                
                # Process subject - use composite key (name, bbox_coordinates)
                if subject_bbox is not None:
                    subject_key = (subject_name, tuple(subject_bbox))
                    if subject_key not in bbox_id_map:
                        bbox_id = f'bbox{bbox_counter}'
                        bbox_id_map[subject_key] = bbox_id
                        bbox_dict[bbox_id] = (subject_name, subject_bbox)
                        bbox_counter += 1
                
                # Process object
                if object_bbox is not None:
                    # Object has bounding box - use composite key
                    object_key = (object_name, tuple(object_bbox))
                    if object_key not in bbox_id_map:
                        bbox_id = f'bbox{bbox_counter}'
                        bbox_id_map[object_key] = bbox_id
                        bbox_dict[bbox_id] = (object_name, object_bbox)
                        bbox_counter += 1
                else:
                    # Object has no bounding box - it's an attribute
                    if object_name not in attr_id_map:
                        attr_id = f'att{attr_counter}'
                        attr_id_map[object_name] = attr_id
                        attr_dict[attr_id] = object_name  # Store attribute name
                        attr_counter += 1
            
            # Create triplets
            triplets = []
            for row in scene_rows:
                _, subject_name, subject_bbox, relationship, object_name, object_bbox = row
                
                if subject_bbox is not None:
                    subject_key = (subject_name, tuple(subject_bbox))
                    subject_id = bbox_id_map[subject_key]
                    
                    if object_bbox is not None:
                        # Object has bbox
                        object_key = (object_name, tuple(object_bbox))
                        object_id = bbox_id_map[object_key]
                    else:
                        # Object is attribute
                        object_id = attr_id_map[object_name]
                    
                    triplets.append([subject_id, relationship, object_id])
            
            scene_graph = SceneGraph(
                triplets=triplets,
                bounding_boxes=bbox_dict,
                attributes=attr_dict  
            )
            
            # Create instances for each query matching this image_id
            matching_queries = [q for q in query_data["queries"] if str(q["image_id"]) == image_id]
            
            for query_item in matching_queries:
                target_obj, target_bbox = query_item["target"]
                
                print(f"\n=== PROCESSING QUERY FOR IMAGE {image_id} ===")
                print(f"Target object: '{target_obj}', Target bbox: {target_bbox}")
                
                # NEW APPROACH: Store target bbox directly in metadata
                # No longer try to find exact bbox_id match
                
                metadata = {
                    'image_id': image_id,
                    'num_objects': len(bbox_id_map),
                    'probability': query_item["probability"],
                    'target_bbox': target_bbox  # NEW: Store target bbox coordinates directly
                }
                
                # Create instance with target_bbox_id = None (since we're using generated scene graphs)
                instance = DatasetInstance(
                    query=query_item["query"],
                    scene_graph=scene_graph,
                    target=(target_obj, None),  # NEW: bbox_id is always None for generated scene graphs
                    metadata=metadata
                )
                
                self.instances.append(instance)
                print(f"Created instance - Target: {instance.target}, Target bbox in metadata: {metadata['target_bbox']}")
            
            print(f"Dataset length after processing image {image_id}: {len(self)}")
            
        print(f"\n=== FINAL DATASET STATISTICS ===")
        print(f"Total instances: {len(self)}")
        print(f"Sample instance target: {self.instances[0].target if self.instances else 'No instances'}")
        print(f"Sample metadata target_bbox: {self.instances[0].metadata.get('target_bbox') if self.instances else 'No instances'}")


class DatasetAnalyzer:
    """
    Helper class to analyze the dataset and ensure proper setup for IoU-based evaluation.
    """
    
    @staticmethod
    def analyze_dataset(dataset: ReferringExpressionDataset):
        """Analyze dataset structure for generated scene graph compatibility"""
        print("\n" + "="*60)
        print("DATASET ANALYSIS FOR GENERATED SCENE GRAPHS")
        print("="*60)
        
        if not dataset.instances:
            print("No instances in dataset!")
            return
        
        # Check target structure
        target_bbox_ids = [inst.target[1] for inst in dataset.instances]
        none_count = sum(1 for tid in target_bbox_ids if tid is None)
        
        print(f"Total instances: {len(dataset.instances)}")
        print(f"Instances with target_bbox_id = None: {none_count} / {len(dataset.instances)}")
        print(f"Properly configured for generated scene graphs: {'✅' if none_count == len(dataset.instances) else '❌'}")
        
        # Check metadata for target_bbox
        has_target_bbox = sum(1 for inst in dataset.instances 
                             if inst.metadata and 'target_bbox' in inst.metadata 
                             and inst.metadata['target_bbox'] is not None)
        
        print(f"Instances with target_bbox in metadata: {has_target_bbox} / {len(dataset.instances)}")
        print(f"Target bbox storage: {'✅' if has_target_bbox == len(dataset.instances) else '❌'}")
        
        # Sample analysis
        sample_inst = dataset.instances[0]
        print(f"\nSample instance:")
        print(f"  Target: {sample_inst.target}")
        print(f"  Target bbox from metadata: {sample_inst.metadata.get('target_bbox')}")
        print(f"  Image ID: {sample_inst.metadata.get('image_id')}")
        print(f"  Scene graph objects: {len(sample_inst.scene_graph.bounding_boxes)}")
        
        # Check for any remaining exact matches (should be zero for generated scene graphs)
        exact_matches = 0
        for inst in dataset.instances:
            target_obj, target_bbox_id = inst.target
            target_bbox_coords = inst.metadata.get('target_bbox')
            
            if target_bbox_coords:
                # Check if any scene graph bbox exactly matches target
                for bbox_id, (obj_name, sg_bbox) in inst.scene_graph.bounding_boxes.items():
                    if obj_name == target_obj and sg_bbox == target_bbox_coords:
                        exact_matches += 1
                        break
        
        print(f"\nExact matches between targets and scene graph: {exact_matches} / {len(dataset.instances)}")
        print(f"Expected for generated scene graphs: 0-{len(dataset.instances) // 10} (some may coincidentally match)")
        
        if exact_matches < len(dataset.instances) * 0.5:  # Less than 50% exact matches
            print("✅ Dataset appears to use generated scene graphs (good for IoU evaluation)")
        else:
            print("⚠️  Dataset may still use ground truth scene graphs (many exact matches)")
        
        print("="*60)