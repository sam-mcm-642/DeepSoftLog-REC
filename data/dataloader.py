from deepsoftlog.data.dataloader import DataLoader
from data.dataset import ReferringExpressionDataset, DatasetInstance, SceneGraph
import random
#from deepsoftlog.data import to_prolog, sg_to_prolog
import pandas as pd
from collections import defaultdict
import csv
from typing import List
import json


class ReferringExpressionDataLoader(DataLoader):
    def __init__(self, dataset: ReferringExpressionDataset, batch_size: int = 1, shuffle: bool = False, seed=None):
        super().__init__(dataset, batch_size, shuffle, seed)
        self.dataset = dataset


    def generate_data_instances(self, f) -> List[DatasetInstance]:
        instances = []
        csv_reader = csv.reader(f)
        
        # Load queries
        with open("data/sg/sample_queries.json", "r") as f_queries:
            query_data = json.load(f_queries)
        
        image_groups = defaultdict(list)
        for row in csv_reader:
            image_id, object_name, object_bbox, relationship, subject_name, subject_bbox = row
            image_groups[image_id].append(row)
        
        for image_id, scene_rows in image_groups.items():
            bbox_dict = {}
            bbox_id_map = {}
            
            bbox_counter = 1
            for row in scene_rows:
                _, object_name, object_bbox, relationship, subject_name, subject_bbox = row
                
                for name, bbox in [(object_name, object_bbox), (subject_name, subject_bbox)]:
                    if name not in bbox_id_map:
                        bbox_id = f'bbox{bbox_counter}'
                        bbox_id_map[name] = bbox_id
                        bbox_dict[bbox_id] = (name, [int(x) for x in bbox.split(',')])
                        bbox_counter += 1
            
            triplets = [
                [bbox_id_map[row[1]], row[3], bbox_id_map[row[4]]] 
                for row in scene_rows
            ]
            
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
                
                instances.append(instance)
        
        return instances
      

            
            
