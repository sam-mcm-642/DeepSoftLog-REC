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
    target: tuple[str, str]  # Target bounding box for the query (man, bbox3)
    metadata: Dict[str, Union[str, int, float]] = None  # Optional metadata like IDs or difficulty level
    
    
class ReferringExpressionDataset(Dataset):
    """
    Custom Dataset for referring expression tasks.
    Extends the abstract Dataset class from DeepSoftLog.
    """
    def __init__(self, instances: List[DatasetInstance] = None):
        self.instances = instances if instances is not None else []
        self._initialized = False  # Track if dataset has been loaded

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
    
    def generate_data_instances(self, filepath, queries_file=None) -> List[DatasetInstance]:
        """
        Generate data instances with improved error handling and validation.
        Only run once per dataset instance.
        """
        if self._initialized:
            print("WARNING: Dataset already initialized. Skipping re-initialization.")
            return
        
        print(f"Initializing dataset from: {filepath}")
        
        # Load queries with fallback
        if queries_file is None:
            queries_file = "/Users/sammcmanagan/Desktop/Thesis/Model/data/query/generated_queries.json"
        
        try:
            with open(queries_file, "r") as f_queries:
                query_data = json.load(f_queries)
            print(f"Loaded {len(query_data.get('queries', []))} queries from {queries_file}")
        except FileNotFoundError:
            print(f"ERROR: Query file not found: {queries_file}")
            return
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in query file: {e}")
            return
        
        image_groups = defaultdict(list)
        
        # Read CSV with proper error handling
        try:
            with open(filepath, 'r') as f:
                csv_reader = csv.reader(f, quotechar='"', escapechar='\\')
                # Skip header
                next(csv_reader)
                
                for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 since we skipped header
                    try:
                        # Validate row length
                        if len(row) < 6:
                            print(f"WARNING: Row {row_num} has insufficient columns: {row}")
                            continue
                        
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
                        
                    except (ValueError, SyntaxError) as e:
                        print(f"WARNING: Error processing row {row_num}: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"ERROR: Scene graph file not found: {filepath}")
            return
        
        print(f"Loaded scene graph data for {len(image_groups)} images")
        
        # Track statistics
        instances_created = 0
        instances_skipped = 0
        target_not_found = 0
        
        # Process each image group
        for image_id, scene_rows in image_groups.items():
            bbox_dict = {}
            bbox_id_map = {}  # Maps (object_name, bbox_tuple) -> bbox_id
            attr_dict = {}    # Maps attribute_name -> att_id
            attr_id_map = {}  # Maps attribute_name -> att_id
            
            bbox_counter = 1
            attr_counter = 1
            
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
                        attr_dict[attr_id] = object_name
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
            matching_queries = [q for q in query_data.get("queries", []) if str(q.get("image_id")) == image_id]
            
            if not matching_queries:
                print(f"No queries found for image_id: {image_id}")
                continue
            
            for query_item in matching_queries:
                try:
                    target_obj, target_bbox = query_item["target"]
                    target_bbox_id = None
                    
                    # Find matching bbox_id for target using composite key
                    target_key = (target_obj, tuple(target_bbox))
                    
                    if target_key in bbox_id_map:
                        target_bbox_id = bbox_id_map[target_key]
                        # print(f"✅ FOUND target {target_obj} at {target_bbox_id}")
                    else:
                        # Fallback: search through bbox_dict for exact match
                        for bbox_id, (obj_name, bbox) in bbox_dict.items():
                            if obj_name == target_obj and bbox == target_bbox:
                                target_bbox_id = bbox_id
                                print(f"✅ FOUND via fallback: {target_bbox_id}")
                                break
                        else:
                            print(f"❌ Target not found: {target_obj} with bbox {target_bbox}")
                            print(f"   Available objects for '{target_obj}': {[key[0] for key in bbox_id_map.keys() if key[0] == target_obj]}")
                            target_not_found += 1
                            instances_skipped += 1
                            continue  # Skip this instance
                    
                    # Only create instance if target was found
                    if target_bbox_id is not None:
                        metadata = {
                            'image_id': image_id,
                            'num_objects': len(bbox_id_map),
                            'probability': query_item.get("probability", 1.0)
                        }
                        
                        instance = DatasetInstance(
                            query=query_item["query"],
                            scene_graph=scene_graph,
                            target=(target_obj, target_bbox_id),
                            metadata=metadata
                        )
                        
                        self.instances.append(instance)
                        instances_created += 1
                    
                except (KeyError, ValueError, TypeError) as e:
                    print(f"ERROR: Invalid query item: {e}")
                    print(f"Query item: {query_item}")
                    instances_skipped += 1
                    continue
        
        # Mark as initialized
        self._initialized = True
        
        # Print final statistics
        print(f"\n=== Dataset Generation Summary ===")
        print(f"Total instances created: {instances_created}")
        print(f"Instances skipped: {instances_skipped}")
        print(f"Targets not found: {target_not_found}")
        print(f"Final dataset length: {len(self.instances)}")
        print(f"Success rate: {instances_created/(instances_created + instances_skipped)*100:.1f}%")
        
        if target_not_found > 0:
            print(f"\nWARNING: {target_not_found} targets could not be matched to scene graph data.")
            print("This might indicate a mismatch between your generated queries and scene graph CSV.")
    
    def validate_dataset(self):
        """Validate dataset integrity"""
        print(f"\n=== Dataset Validation ===")
        
        invalid_instances = []
        for i, instance in enumerate(self.instances):
            # Check if target bbox_id exists in scene graph
            target_obj, target_bbox_id = instance.target
            if target_bbox_id not in instance.scene_graph.bounding_boxes:
                invalid_instances.append(i)
                print(f"INVALID: Instance {i} has target_bbox_id '{target_bbox_id}' not in scene graph")
        
        if invalid_instances:
            print(f"Found {len(invalid_instances)} invalid instances")
            # Optionally remove invalid instances
            # for i in reversed(invalid_instances):
            #     del self.instances[i]
        else:
            print("All instances are valid!")
        
        # Print dataset statistics
        image_ids = set(inst.metadata['image_id'] for inst in self.instances)
        target_objects = [inst.target[0] for inst in self.instances]
        
        print(f"Unique images: {len(image_ids)}")
        print(f"Most common targets: {pd.Series(target_objects).value_counts().head()}")