import pandas as pd
import json
import random
import ast
from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np
import os
from tqdm import tqdm

class QueryGenerator:
    def __init__(self, scene_graph_path: str, ontology_path: str, 
                 ontology_mutation_prob: float = 0.3,
                 min_expr_predicates: int = 1, 
                 max_expr_predicates: int = 4,
                 chunk_size: int = 10000):
        """
        Initialize the query generator with scene graph and ontology data.
        
        Args:
            scene_graph_path: Path to the scene graph CSV or JSON
            ontology_path: Path to the ontology CSV
            ontology_mutation_prob: Probability of mutating a term using the ontology
            min_expr_predicates: Minimum number of expression predicates per query
            max_expr_predicates: Maximum number of expression predicates per query
            chunk_size: Chunk size for reading large CSV files
        """
        self.scene_graph_path = scene_graph_path
        self.chunk_size = chunk_size
        
        # Check if the scene graph file is JSON or CSV
        if scene_graph_path.endswith('.json'):
            # Load only the necessary metadata now, load chunks later
            print(f"Loading scene graph metadata from JSON: {scene_graph_path}")
            with open(scene_graph_path, 'r') as f:
                # Just read the header or first few records to get metadata
                sample_data = json.load(f, object_hook=lambda d: {k: d[k] for k in list(d)[:10]})
            self.is_json = True
            self.scene_graph_df = None  # Will be loaded in chunks during processing
        else:
            # For CSV, we'll read it in chunks during processing
            print(f"Scene graph will be loaded from CSV in chunks: {scene_graph_path}")
            # Just read the header to get column names
            self.scene_graph_df = pd.read_csv(scene_graph_path, nrows=0)
            self.is_json = False
            
        # Load the ontology - this is typically much smaller
        print(f"Loading ontology from: {ontology_path}")
        self.ontology_df = pd.read_csv(ontology_path)
        
        self.ontology_mutation_prob = ontology_mutation_prob
        self.min_expr_predicates = min_expr_predicates
        self.max_expr_predicates = max_expr_predicates
        
        # Build ontology mappings
        self.build_ontology_mappings()
        
    def build_ontology_mappings(self):
        """Build useful mappings from the ontology for quick lookup."""
        # Mapping from term to related terms through ontology relations
        self.term_relations = {}
        
        for _, row in self.ontology_df.iterrows():
            subject = row['subject']
            relation = row['relation']
            obj = row['object']
            
            # Initialize if not exists
            if subject not in self.term_relations:
                self.term_relations[subject] = []
            
            self.term_relations[subject].append((relation, obj))
            
            # For symmetric relations like synonym, add the reverse mapping
            if relation == 'synonym':
                if obj not in self.term_relations:
                    self.term_relations[obj] = []
                self.term_relations[obj].append((relation, subject))
        
        # Build hyponym/hypernym mappings for type predicate generation
        self.hyponym_to_hypernym = {}
        hyponym_rows = self.ontology_df[self.ontology_df['relation'] == 'hyponym']
        
        for _, row in hyponym_rows.iterrows():
            self.hyponym_to_hypernym[row['subject']] = row['object']
            
        # Also add synonym mappings to the hyponym/hypernym mappings
        synonym_rows = self.ontology_df[self.ontology_df['relation'] == 'synonym']
        for _, row in synonym_rows.iterrows():
            if row['subject'] in self.hyponym_to_hypernym:
                self.hyponym_to_hypernym[row['object']] = self.hyponym_to_hypernym[row['subject']]
            if row['object'] in self.hyponym_to_hypernym:
                self.hyponym_to_hypernym[row['subject']] = self.hyponym_to_hypernym[row['object']]
    
    def get_related_term(self, term: str, relation_type: Optional[str] = None) -> Optional[str]:
        """
        Get a term related to the input term from the ontology.
        
        Args:
            term: The term to find related terms for
            relation_type: If specified, only return terms with this relation type
            
        Returns:
            A related term if found, None otherwise
        """
        if term not in self.term_relations:
            return None
        
        related_terms = self.term_relations[term]
        
        if relation_type:
            related_terms = [t for t in related_terms if t[0] == relation_type]
        
        if not related_terms:
            return None
            
        # Randomly select one related term
        relation, related_term = random.choice(related_terms)
        return related_term
    
    def get_hypernym(self, term: str) -> str:
        """
        Get the hypernym of a term from the ontology if it exists,
        otherwise return the term itself.
        """
        return self.hyponym_to_hypernym.get(term, term)
    
    def find_valid_targets_in_image(self, image_id: int) -> List[Tuple[str, List[int]]]:
        """
        Find valid target objects in the given image.
        A valid target must have a bounding box and be part of at least one triplet.
        
        Args:
            image_id: The ID of the image to find targets in
            
        Returns:
            List of tuples (object_name, bounding_box)
        """
        if self.scene_graph_df is None:
            # Load data for this image if not already loaded
            self.scene_graph_df = self._load_scene_graph_chunk([image_id])
            
            if self.scene_graph_df.empty:
                print(f"No scene graph data found for image ID {image_id}")
                return []
                
        # Filter scene graph for the specific image
        image_sg = self.scene_graph_df[self.scene_graph_df['image_id'] == image_id]
        
        if image_sg.empty:
            print(f"Image ID {image_id} not found in loaded scene graph data")
            return []
        
        valid_targets = []
        
        # Consider subjects with bounding boxes
        subject_candidates = image_sg[image_sg['subject_bbox'].notna()]
        for _, row in subject_candidates.iterrows():
            subject = row['subject']
            bbox = row['subject_bbox']
            
            # Only add if it's not already in the list
            if bbox and (subject, bbox) not in valid_targets:
                valid_targets.append((subject, bbox))
        
        # Consider objects with bounding boxes
        object_candidates = image_sg[image_sg['object_bbox'].notna()]
        for _, row in object_candidates.iterrows():
            obj = row['object']
            bbox = row['object_bbox']
            
            # Only add if it's not already in the list and not NULL
            if bbox and obj != 'NULL' and (obj, bbox) not in valid_targets:
                valid_targets.append((obj, bbox))
        
        return valid_targets
    
    def find_triplets_for_target(self, image_id: int, target: str, target_bbox: List[int]) -> List[Dict[str, Any]]:
        """
        Find all triplets in the scene graph that involve the target.
        
        Args:
            image_id: The image ID
            target: The target object name
            target_bbox: The target bounding box
            
        Returns:
            List of dictionaries representing triplets
        """
        image_sg = self.scene_graph_df[self.scene_graph_df['image_id'] == image_id]
        triplets = []
        
        # Check for triplets where the target is the subject
        subject_triplets = image_sg[
            (image_sg['subject'] == target) & 
            (image_sg['subject_bbox'].apply(lambda x: x == target_bbox if x else False))
        ]
        
        for _, row in subject_triplets.iterrows():
            triplets.append({
                'relation': row['relationship'],
                'position': 'subject',
                'other': row['object'],
                'other_bbox': row['object_bbox']
            })
        
        # Check for triplets where the target is the object
        object_triplets = image_sg[
            (image_sg['object'] == target) & 
            (image_sg['object_bbox'].apply(lambda x: x == target_bbox if x else False))
        ]
        
        for _, row in object_triplets.iterrows():
            triplets.append({
                'relation': row['relationship'],
                'position': 'object',
                'other': row['subject'],
                'other_bbox': row['subject_bbox']
            })
            
        return triplets
    
    def potentially_mutate_term(self, term: str) -> str:
        """
        Potentially mutate a term using the ontology based on the mutation probability.
        
        Args:
            term: The term to potentially mutate
            
        Returns:
            Either the original term or a related term
        """
        if random.random() < self.ontology_mutation_prob and term in self.term_relations:
            related_term = self.get_related_term(term)
            if related_term:
                return related_term
        return term
    
    def generate_expression_predicate(self, triplet: Dict[str, Any]) -> str:
        """
        Generate an expression predicate from a triplet, potentially mutating terms.
        
        Args:
            triplet: Dictionary containing relation, position, other, other_bbox
            
        Returns:
            Expression predicate as a string
        """
        relation = self.potentially_mutate_term(triplet['relation'])
        other = triplet['other']
        
        # Skip if other is NULL or other doesn't have a bbox when needed
        if other == 'NULL' or (triplet['other_bbox'] is None and triplet['relation'] != 'hasAttribute'):
            return None
            
        # Potentially mutate the other term
        other = self.potentially_mutate_term(other)
        
        if triplet['position'] == 'subject':
            # Target is subject, so expression is (relation, X, other)
            return f"expression({relation}, X, {other})"
        else:
            # Target is object, so expression is (relation, other, X)
            return f"expression({relation}, {other}, X)"
    
    def generate_query_for_image(self, image_id: int, num_queries: int = 1) -> List[Dict[str, Any]]:
        """
        Generate queries for a specific image.
        
        Args:
            image_id: The image ID
            num_queries: Number of queries to generate
            
        Returns:
            List of query dictionaries
        """
        queries = []
        valid_targets = self.find_valid_targets_in_image(image_id)
        
        if not valid_targets:
            return []
            
        # Try to generate the requested number of queries
        attempts = 0
        max_attempts = num_queries * 3  # Allow more attempts than needed queries
        
        while len(queries) < num_queries and attempts < max_attempts:
            attempts += 1
            
            # Randomly select a target
            target, target_bbox = random.choice(valid_targets)
            
            # Find triplets involving the target
            triplets = self.find_triplets_for_target(image_id, target, target_bbox)
            
            # Skip if no valid triplets
            if not triplets:
                continue
                
            # Select number of expression predicates
            num_expr_predicates = random.randint(self.min_expr_predicates, 
                                                min(self.max_expr_predicates, len(triplets)))
            
            # Randomly select triplets
            selected_triplets = random.sample(triplets, num_expr_predicates)
            
            # Generate expression predicates
            expr_predicates = []
            for triplet in selected_triplets:
                predicate = self.generate_expression_predicate(triplet)
                if predicate:
                    expr_predicates.append(predicate)
            
            # Skip if no valid expression predicates
            if not expr_predicates:
                continue
                
            # Get the hypernym for the target object for the type predicate
            target_type = self.get_hypernym(target)
            
            # Construct the query
            query_text = f"target(X), type(X, {target_type}), {', '.join(expr_predicates)}"
            
            query_obj = {
                "image_id": image_id,
                "query": query_text,
                "target": [target, target_bbox],
                "probability": 1.0
            }
            
            queries.append(query_obj)
            
        return queries
    
    def convert_numpy_types(self, obj):
        """
        Convert NumPy types to native Python types for JSON serialization.
        
        Args:
            obj: The object to convert
            
        Returns:
            The object with NumPy types converted to native Python types
        """
        if isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self.convert_numpy_types(obj.tolist())
        else:
            return obj
    
    def _load_scene_graph_chunk(self, image_ids=None):
        """
        Load a chunk of the scene graph data.
        
        Args:
            image_ids: Optional list of specific image IDs to load
            
        Returns:
            DataFrame containing the loaded chunk
        """
        if self.is_json:
            # Load relevant portion from JSON file
            with open(self.scene_graph_path, 'r') as f:
                data = json.load(f)
                
            # If specific image IDs are requested, filter the data
            if image_ids:
                filtered_data = [item for item in data if item.get('image_id') in image_ids]
                chunk_df = pd.DataFrame(filtered_data)
            else:
                # Load a chunk
                chunk_df = pd.DataFrame(data[:self.chunk_size])
        else:
            # For CSV, use pandas chunk reading
            if image_ids:
                # This is less efficient for CSVs, but for targeted image IDs we need to scan the file
                chunks = []
                for chunk in pd.read_csv(self.scene_graph_path, chunksize=self.chunk_size):
                    filtered_chunk = chunk[chunk['image_id'].isin(image_ids)]
                    if not filtered_chunk.empty:
                        chunks.append(filtered_chunk)
                chunk_df = pd.concat(chunks) if chunks else pd.DataFrame()
            else:
                # Just read the next chunk
                chunk_df = pd.read_csv(self.scene_graph_path, chunksize=self.chunk_size).get_chunk()
                
        # Process bounding boxes in the chunk
        if not chunk_df.empty:
            chunk_df['subject_bbox'] = chunk_df['subject_bbox'].apply(
                lambda x: ast.literal_eval(x) if x != 'NULL' and pd.notna(x) else None
            )
            chunk_df['object_bbox'] = chunk_df['object_bbox'].apply(
                lambda x: ast.literal_eval(x) if x != 'NULL' and pd.notna(x) else None
            )
            
        return chunk_df
    
    def get_unique_image_ids(self, max_images=None):
        """
        Get unique image IDs from the scene graph.
        
        Args:
            max_images: Maximum number of image IDs to return
            
        Returns:
            List of unique image IDs
        """
        if self.is_json:
            # For JSON, we need to scan the file to get unique IDs
            with open(self.scene_graph_path, 'r') as f:
                data = json.load(f)
            
            # Extract unique image IDs
            image_ids = set(item.get('image_id') for item in data if 'image_id' in item)
            image_ids = list(image_ids)
        else:
            # For CSV, we read in chunks to get unique IDs
            unique_ids = set()
            for chunk in pd.read_csv(self.scene_graph_path, usecols=['image_id'], chunksize=self.chunk_size):
                unique_ids.update(chunk['image_id'].unique())
                if max_images and len(unique_ids) >= max_images:
                    break
            
            image_ids = list(unique_ids)
        
        # Limit to max_images if specified
        if max_images:
            image_ids = image_ids[:max_images]
            
        return image_ids
        
    def generate_queries_in_batches(self, output_path: str, batch_size: int = 100, 
                                   queries_per_image: Tuple[int, int] = (1, 3),
                                   resume: bool = False, save_interval: int = 10,
                                   max_images: Optional[int] = None):
        """
        Generate queries in batches and save incrementally to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
            batch_size: Number of images to process in each batch
            queries_per_image: Tuple (min, max) of queries to generate per image
            resume: Whether to resume from an existing file
            save_interval: Save after processing this many images
            max_images: Maximum number of images to process (None for all)
        """
        # Get unique image IDs, optionally limited to max_images
        print("Getting unique image IDs...")
        image_ids = self.get_unique_image_ids(max_images)
        print(f"Found {len(image_ids)} unique image IDs")
        
        # Check if we need to resume from an existing file
        existing_queries = []
        processed_image_ids = set()
        
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
                    existing_queries = existing_data.get('queries', [])
                    
                    # Extract processed image IDs
                    for query in existing_queries:
                        processed_image_ids.add(query['image_id'])
                        
                print(f"Resuming from existing file with {len(existing_queries)} queries and {len(processed_image_ids)} processed images.")
            except Exception as e:
                print(f"Error loading existing file: {e}. Starting fresh.")
                existing_queries = []
                processed_image_ids = set()
        
        # Filter out already processed image IDs
        remaining_image_ids = [img_id for img_id in image_ids if img_id not in processed_image_ids]
        print(f"Remaining images to process: {len(remaining_image_ids)}")
        
        # Process in batches
        all_queries = existing_queries
        processed_count = 0
        
        for i in range(0, len(remaining_image_ids), batch_size):
            batch_image_ids = remaining_image_ids[i:i+batch_size]
            batch_queries = []
            
            print(f"Processing batch {i//batch_size + 1} with {len(batch_image_ids)} images...")
            
            # Load the scene graph data for this batch of image IDs
            print("Loading scene graph data for this batch...")
            self.scene_graph_df = self._load_scene_graph_chunk(batch_image_ids)
            
            # Skip if no data was found for these image IDs
            if self.scene_graph_df.empty:
                print(f"No scene graph data found for batch {i//batch_size + 1}. Skipping.")
                continue
                
            for image_id in tqdm(batch_image_ids):
                # Check if this image ID exists in the loaded data
                if image_id not in self.scene_graph_df['image_id'].values:
                    continue
                    
                num_queries = random.randint(queries_per_image[0], queries_per_image[1])
                image_queries = self.generate_query_for_image(image_id, num_queries)
                batch_queries.extend(image_queries)
                processed_count += 1
                
                # Save at specified intervals
                if processed_count % save_interval == 0:
                    all_queries.extend(batch_queries)
                    self.save_queries(output_path, all_queries)
                    batch_queries = []  # Reset batch queries after saving
            
            # Add any remaining batch queries
            if batch_queries:
                all_queries.extend(batch_queries)
                self.save_queries(output_path, all_queries)
            
            print(f"Batch completed. Total queries so far: {len(all_queries)}")
            
            # Clear the dataframe to free memory
            self.scene_graph_df = None
        
        # Final save
        self.save_queries(output_path, all_queries)
        print(f"All done! Generated a total of {len(all_queries)} queries and saved to {output_path}")
    
    def save_queries(self, output_path: str, queries: List[Dict[str, Any]]):
        """
        Save queries to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
            queries: List of query dictionaries
        """
        # Convert NumPy types to native Python types
        queries_copy = self.convert_numpy_types(queries)
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Wrap in the expected format
        queries_data = {"queries": queries_copy}
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(queries_data, f, indent=4)
        
        print(f"Saved {len(queries)} queries to {output_path}")
    
    def generate_all_queries(self, queries_per_image: Tuple[int, int] = (1, 3)) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate queries for all images in the scene graph.
        
        Args:
            queries_per_image: Tuple (min, max) of queries to generate per image
            
        Returns:
            Dictionary with a "queries" key containing a list of query objects
        """
        all_queries = []
        
        # Get unique image IDs
        image_ids = self.scene_graph_df['image_id'].unique()
        
        for image_id in image_ids:
            num_queries = random.randint(queries_per_image[0], queries_per_image[1])
            image_queries = self.generate_query_for_image(image_id, num_queries)
            all_queries.extend(image_queries)
            
        return {"queries": all_queries}
    
    def save_queries_to_json(self, output_path: str, queries_per_image: Tuple[int, int] = (1, 3)):
        """
        Generate queries and save them to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
            queries_per_image: Tuple (min, max) of queries to generate per image
        """
        queries = self.generate_all_queries(queries_per_image)
        
        # Convert NumPy types to native Python types
        queries = self.convert_numpy_types(queries)
        
        with open(output_path, 'w') as f:
            json.dump(queries, f, indent=4)
        
        print(f"Generated {len(queries['queries'])} queries and saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Paths to your data
    scene_graph_path = "/Users/sammcmanagan/Desktop/Thesis/Model/data/vg/scene_graphs.json"
    ontology_path = "data/wordnet_vg_relationships_readable_s50_o50.csv"
    output_path = "data/query/generated_queries.json"
    
    # Initialize the query generator with chunk size appropriate for your system memory
    generator = QueryGenerator(
        scene_graph_path=scene_graph_path,
        ontology_path=ontology_path,
        ontology_mutation_prob=0.3,
        min_expr_predicates=1,
        max_expr_predicates=4,
        chunk_size=10000  # Adjust based on your memory constraints
    )
    
    # Use the new batch processing method with a limit on the number of images
    generator.generate_queries_in_batches(
        output_path=output_path,
        batch_size=50,            # Process 50 images at a time
        queries_per_image=(1, 3), # Generate 1-3 queries per image
        resume=True,              # Resume from existing file if it exists
        save_interval=10,         # Save after processing 10 images
        max_images=1000           # Only process the first 1000 images
    )
    
    # Or use the original method
    # generator.save_queries_to_json(
    #     output_path=output_path,
    #     queries_per_image=(1, 3)
    # )