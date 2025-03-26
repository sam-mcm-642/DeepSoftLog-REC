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
        
        # Load queries
        with open("data/program/sample_queries.json", "r") as f_queries:
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
                object_bbox = ast.literal_eval(row[5]) if row[5] != 'NULL' else None
                

                processed_row = [
                    str(row[0]),  # image_id
                    row[1],       # subject
                    subject_bbox,
                    row[3],       # relationship
                    row[4],       # object
                    object_bbox
                ]
                print(processed_row)
                
                image_groups[processed_row[0]].append(processed_row)
        
        # Process each image group
        for image_id, scene_rows in image_groups.items():
            bbox_dict = {}
            bbox_id_map = {}
            
            bbox_counter = 1
            for row in scene_rows:
                _, subject_name, subject_bbox, relationship, object_name, object_bbox = row
                
                # Process subject
                if subject_name not in bbox_id_map:
                    bbox_id = f'bbox{bbox_counter}'
                    bbox_id_map[subject_name] = bbox_id
                    bbox_dict[bbox_id] = (subject_name, subject_bbox)
                    bbox_counter += 1
                
                # Process object if it has a bounding box
                if object_bbox is not None and object_name not in bbox_id_map:
                    bbox_id = f'bbox{bbox_counter}'
                    bbox_id_map[object_name] = bbox_id
                    bbox_dict[bbox_id] = (object_name, object_bbox)
                    bbox_counter += 1
            
            # Create triplets using bbox_id_map
            triplets = []
            for row in scene_rows:
                subject_id = bbox_id_map[row[1]]
                relationship = row[3]
                # Only add triplet if object has a bounding box
                if row[5] is not None:
                    object_id = bbox_id_map[row[4]]
                    triplets.append([subject_id, relationship, object_id])
            
            scene_graph = SceneGraph(
                triplets=triplets,
                bounding_boxes=bbox_dict
            )
            
            # Create instances for each query matching this image_id
            matching_queries = [q for q in query_data["queries"] if str(q["image_id"]) == image_id]
            
            for query_item in matching_queries:
                target_obj, target_bbox = query_item["target"]
                target_bbox_id = None
                
                # Find matching bbox_id for target
                for bbox_id, (obj_name, bbox) in bbox_dict.items():
                    if obj_name == target_obj and bbox == target_bbox:
                        target_bbox_id = bbox_id
                        break
                
                metadata = {
                    'image_id': image_id,
                    'num_objects': len(bbox_id_map),
                    'probability': query_item["probability"]
                }
                
                instance = DatasetInstance(
                    query=query_item["query"],
                    scene_graph=scene_graph,
                    target=(target_obj, target_bbox_id),
                    metadata=metadata
                )
                
                self.instances.append(instance)
                print(f"Dataset length: {len(self)}")
        
        #return instances