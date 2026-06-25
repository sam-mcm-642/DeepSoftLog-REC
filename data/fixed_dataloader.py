from deepsoftlog.data.dataloader import DataLoader
from data.fixed_dataset import ReferringExpressionDataset, DatasetInstance, SceneGraph
import random
import pandas as pd
from collections import defaultdict
import csv
from typing import List
import json


class ReferringExpressionDataLoader(DataLoader):
    def __init__(self, dataset: ReferringExpressionDataset, batch_size: int = 1, shuffle: bool = False, seed=None):
        super().__init__(dataset, batch_size, shuffle, seed)
        self.dataset = dataset

    @staticmethod 
    def create_dataset_from_files(scene_graph_file: str, queries_file: str = None, max_instances: int = None):
        """
        Static method to create dataset from files with proper initialization.
        This ensures the dataset is only initialized once.
        """
        print(f"Creating dataset from:")
        print(f"  Scene graph: {scene_graph_file}")
        print(f"  Queries: {queries_file}")
        
        # Create empty dataset
        dataset = ReferringExpressionDataset()
        
        # Generate data instances (only once)
        dataset.generate_data_instances(scene_graph_file, queries_file)
        
        # Validate dataset
        dataset.validate_dataset()
        
        # Limit instances if specified
        if max_instances and max_instances < len(dataset.instances):
            print(f"Limiting dataset to {max_instances} instances (out of {len(dataset.instances)})")
            dataset.instances = dataset.instances[:max_instances]
        
        return dataset

    def generate_data_instances(self, f) -> List[DatasetInstance]:
        """
        DEPRECATED: Use create_dataset_from_files instead.
        This method is kept for backwards compatibility but should not be used.
        """
        print("WARNING: generate_data_instances is deprecated. Use create_dataset_from_files instead.")
        return []