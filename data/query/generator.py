import os
import json
import random
import ast
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import Counter, defaultdict
from tqdm import tqdm

class QueryGenerator:
    def __init__(self, scene_graph_path: str, ontology_path: str, 
                 ontology_mutation_prob: float = 0.3,
                 min_expr_predicates: int = 1, 
                 max_expr_predicates: int = 5,
                 chunk_size: int = 100,
                 negative_ratio: float = 0.3,
                 min_term_usage: int = 10):  # New parameter for minimum term usage
        """
        Initialize the query generator with scene graph and ontology data.
        
        Args:
            scene_graph_path: Path to the scene graph CSV or JSON
            ontology_path: Path to the ontology CSV
            ontology_mutation_prob: Probability of mutating a term using the ontology
            min_expr_predicates: Minimum number of expression predicates per query
            max_expr_predicates: Maximum number of expression predicates per query
            chunk_size: Chunk size for reading large CSV files
            negative_ratio: Ratio of negative instances to generate (0.0 to 1.0)
            min_term_usage: Minimum number of times each ontology term should be used
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("query_generator.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("QueryGenerator")
        
        self.scene_graph_path = scene_graph_path
        self.chunk_size = chunk_size
        self.negative_ratio = negative_ratio
        self.min_term_usage = min_term_usage
        
        # Check if the scene graph file is JSON or CSV
        if scene_graph_path.endswith('.json'):
            self.logger.info(f"Loading scene graph metadata from JSON: {scene_graph_path}")
            with open(scene_graph_path, 'r') as f:
                # Just read the header or first few records to get metadata
                sample_data = json.load(f, object_hook=lambda d: {k: d[k] for k in list(d)[:10]})
            self.is_json = True
            self.scene_graph_df = None  # Will be loaded in chunks during processing
        else:
            self.logger.info(f"Scene graph will be loaded from CSV in chunks: {scene_graph_path}")
            # Just read the header to get column names
            self.scene_graph_df = pd.read_csv(scene_graph_path, nrows=0)
            self.is_json = False
            
        # Load the ontology - this is typically much smaller
        self.logger.info(f"Loading ontology from: {ontology_path}")
        self.ontology_df = pd.read_csv(ontology_path)
        
        self.ontology_mutation_prob = ontology_mutation_prob
        self.min_expr_predicates = min_expr_predicates
        self.max_expr_predicates = max_expr_predicates
        
        # Add caches for performance
        self.triplet_cache = {}  # Cache triplets by (image_id, target, bbox)
        
        # NEW: Ontology term usage tracking
        self.term_usage_count = Counter()  # Track how many times each term has been swapped
        self.original_term_to_swapped = defaultdict(set)  # Track what terms were swapped to what
        self.underused_terms = set()  # Terms that need more usage
        
        # Build ontology mappings
        self.build_ontology_mappings()
        
        # Initialize semantic categories
        self.initialize_semantic_categories()
        
        # Initialize term tracking
        self.initialize_term_tracking()
        
        # Analyze the ontology for statistics
        self.analyze_ontology_relationships()
    
    def initialize_term_tracking(self):
        """Initialize tracking for ontology term usage."""
        # Extract all terms from ontology
        all_terms = set()
        all_terms.update(self.ontology_df['subject'].unique())
        all_terms.update(self.ontology_df['object'].unique())
        
        # Initialize usage counter for all terms
        for term in all_terms:
            self.term_usage_count[term] = 0
        
        # All terms start as underused
        self.underused_terms = all_terms.copy()
        
        self.logger.info(f"Initialized tracking for {len(all_terms)} ontology terms")
    
    def update_term_usage(self, original_term: str, swapped_term: str):
        """Update usage tracking when a term is swapped."""
        if original_term != swapped_term:
            self.term_usage_count[original_term] += 1
            self.term_usage_count[swapped_term] += 1
            self.original_term_to_swapped[original_term].add(swapped_term)
            
            # Remove from underused if they've reached minimum usage
            if self.term_usage_count[original_term] >= self.min_term_usage:
                self.underused_terms.discard(original_term)
            if self.term_usage_count[swapped_term] >= self.min_term_usage:
                self.underused_terms.discard(swapped_term)
    
    def get_underused_terms_report(self) -> Dict[str, Any]:
        """Get a report on term usage statistics."""
        total_terms = len(self.term_usage_count)
        underused_count = len(self.underused_terms)
        
        # Get terms sorted by usage count
        sorted_by_usage = self.term_usage_count.most_common()
        least_used = sorted_by_usage[-10:] if len(sorted_by_usage) >= 10 else sorted_by_usage
        most_used = sorted_by_usage[:10]
        
        return {
            'total_terms': total_terms,
            'underused_terms_count': underused_count,
            'underused_percentage': (underused_count / total_terms * 100) if total_terms > 0 else 0,
            'underused_terms': sorted(list(self.underused_terms)),
            'least_used_terms': least_used,
            'most_used_terms': most_used,
            'terms_with_sufficient_usage': total_terms - underused_count
        }
    
    def save_term_usage_report(self, output_path: str):
        """Save term usage report to a JSON file."""
        report = self.get_underused_terms_report()
        
        # Add detailed mapping information
        report['term_swap_mappings'] = {
            term: list(swapped_set) for term, swapped_set in self.original_term_to_swapped.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Term usage report saved to: {output_path}")
        self.logger.info(f"Underused terms: {report['underused_terms_count']}/{report['total_terms']} "
                        f"({report['underused_percentage']:.1f}%)")

    def initialize_semantic_categories(self):
        """Create categories of semantically compatible terms"""
        # Extract terms from the ontology
        all_terms = set(self.ontology_df['subject'].unique()) | set(self.ontology_df['object'].unique())
        
        # Define basic categories based on hypernym relationships
        self.semantic_categories = defaultdict(set)
        
        # Find top-level categories (common hypernyms)
        hypernym_rows = self.ontology_df[self.ontology_df['relation'] == 'hypernym']
        
        top_hypernyms = Counter()
        for _, row in hypernym_rows.iterrows():
            top_hypernyms[row['object']] += 1
        
        # Use top hypernyms as categories
        for hypernym, count in top_hypernyms.most_common(20):
            if count >= 5:  # Only use categories with enough members
                category_name = hypernym.replace('_', ' ')
                
                # Find all hyponyms of this hypernym
                for _, row in hypernym_rows.iterrows():
                    if row['object'] == hypernym:
                        self.semantic_categories[category_name].add(row['subject'])
                
                # Add the hypernym itself
                self.semantic_categories[category_name].add(hypernym)
        
        # Build reverse mapping
        self.term_to_category = {}
        for category, terms in self.semantic_categories.items():
            for term in terms:
                self.term_to_category[term] = category
        
        self.logger.info(f"Created {len(self.semantic_categories)} semantic categories")
        
    def analyze_ontology_relationships(self):
        """Analyze ontology relationships for quality and coverage"""
        relation_counts = Counter()
        term_counts = Counter()
        
        # Count by relation type
        for _, row in self.ontology_df.iterrows():
            relation_counts[row['relation']] += 1
            term_counts[row['subject']] += 1
            term_counts[row['object']] += 1
        
        # Log statistics
        self.logger.info("=== Ontology Relationship Statistics ===")
        self.logger.info(f"Total relationships: {len(self.ontology_df)}")
        self.logger.info("Relationship types:")
        for rel, count in relation_counts.most_common():
            self.logger.info(f"- {rel}: {count} ({count/len(self.ontology_df)*100:.1f}%)")
        
        self.logger.info("Most connected terms:")
        for term, count in term_counts.most_common(10):
            self.logger.info(f"- {term}: {count} connections")
        
        # Check if we have good synonym coverage
        total_terms = len(set(term_counts.keys()))
        terms_with_synonyms = sum(1 for t in term_counts if 
                                 any(r[0] == 'synonym' for r in self.term_relations.get(t, [])))
        
        self.logger.info(f"Terms with synonyms: {terms_with_synonyms}/{total_terms} ({terms_with_synonyms/total_terms*100:.1f}%)")
        
        # Create a list of terms that are safe for hyponym relationships
        self.safe_for_hyponyms = set()
        for term in term_counts:
            # Look for terms that have multiple hyponyms
            hyponyms = [t for r, t in self.term_relations.get(term, []) if r == 'hyponym']
            if len(hyponyms) >= 3:
                self.safe_for_hyponyms.add(term)
    
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
        # NEW - look for correct relation type  
        hypernym_rows = self.ontology_df[self.ontology_df['relation'] == 'hypernym']

        for _, row in hypernym_rows.iterrows():
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
        Get a term related to the input term from the ontology with priority weighting.
        
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
                
            # Randomly select one related term with the specified relation type
            _, related_term = random.choice(related_terms)
            return related_term
        
        # Define relationship priorities (higher = more likely to use)
        relation_weights = {
            'synonym': 10.0,     # Highest priority - same meaning
            'hyponym': 3.0,      # Medium priority - more specific
            'hypernym': 2.0,     # Lower priority - more general
            'part_meronym': 1.0, # Low priority - parts
            'part_holonym': 1.0  # Low priority - wholes
        }
        
        # Group relations by type
        by_relation_type = defaultdict(list)
        for relation_type, related_term in related_terms:
            by_relation_type[relation_type].append(related_term)
        
        # Choose relation type with weighted probability
        relation_types = list(by_relation_type.keys())
        weights = [relation_weights.get(rel, 0.5) for rel in relation_types]
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]
        else:
            return None
        
        # Select a relation type based on weights
        chosen_relation = random.choices(relation_types, weights=weights, k=1)[0]
        
        # Now randomly select a term of this relation type
        return random.choice(by_relation_type[chosen_relation])
    
    def get_hypernym(self, term: str) -> str:
        """
        Get the hypernym of a term from the ontology if it exists,
        otherwise return the term itself.
        """
        # Try direct lookup
        if term in self.hyponym_to_hypernym:
            return self.hyponym_to_hypernym[term]
        
        # Try normalized versions
        normalized_term = term.replace('-', ' ').replace('_', ' ')
        if normalized_term in self.hyponym_to_hypernym:
            return self.hyponym_to_hypernym[normalized_term]
        
        # No hypernym found, return original
        return term
    
    def apply_term_relationship(self, term: str, context: str) -> str:
        """
        Apply appropriate relations based on context:
        - In type predicates: prefer hypernyms (more general)
        - In expression predicates: prefer synonyms, limit others
        
        Args:
            term: The term to apply a relationship to
            context: The context in which the term is used ("type_predicate", 
                    "expression_predicate", "relation_predicate")
                    
        Returns:
            A related term based on context, or the original term if no suitable related term is found
        """
        if term not in self.term_relations:
            return term
            
        if context == "type_predicate":
            # For type predicates, ONLY go more general (dog → animal)
            hypernyms = [t for r, t in self.term_relations[term] if r == 'hypernym']
            if hypernyms and random.random() < 0.7:  # 70% chance to use hypernym
                return random.choice(hypernyms)
        
        elif context == "expression_predicate":
            # For expression predicates, predominantly use synonyms
            synonyms = [t for r, t in self.term_relations[term] if r == 'synonym']
            
            if synonyms and random.random() < 0.8:  # 80% chance to use synonym
                return random.choice(synonyms)
            
            # Occasionally use careful selections of other relations
            safe_terms = []
            
            # Add hyponyms (more specific) with lower probability
            # Only for certain term types that make sense
            if term in self.safe_for_hyponyms and random.random() < 1.0:
                hyponyms = [t for r, t in self.term_relations[term] if r == 'hyponym']
                safe_terms.extend(hyponyms)
                
            if safe_terms:
                return random.choice(safe_terms)
        
        elif context == "relation_predicate":
            # For relation predicates, predominantly use synonyms
            synonyms = [t for r, t in self.term_relations[term] if r == 'synonym']
            
            if synonyms and random.random() < 0.9:  # 90% chance to use synonym for relations
                return random.choice(synonyms)
        
        # Default: return original term
        return term
    
    def potentially_mutate_term(self, term: str, context: str = "expression_predicate") -> str:
        """
        Potentially mutate a term using the ontology based on the mutation probability.
        Preserves semantic category whenever possible.
        
        Args:
            term: The term to potentially mutate
            context: The context in which the term is used
            
        Returns:
            Either the original term or a related term
        """
        # Clean and normalize term
        if not isinstance(term, str):
            term = str(term)
        
        term = term.strip()
        
        # Return original term if it's empty or invalid
        if len(term) == 0 or term == 'NULL':
            return term
        
        if random.random() < self.ontology_mutation_prob:
            # Get the category of this term if it exists
            category = self.term_to_category.get(term)
            
            # Get the related term using context-specific approach
            related_term = self.apply_term_relationship(term, context)
            
            # Validate the related term
            if (related_term and isinstance(related_term, str) and 
                len(related_term.strip()) > 0 and related_term != 'NULL'):
                
                # If we have a category, verify the related term is in same category
                if category and related_term in self.term_to_category:
                    if self.term_to_category[related_term] == category:
                        # UPDATE USAGE TRACKING
                        self.update_term_usage(term, related_term)
                        return related_term
                    # If not in same category, fall through to use original
                else:
                    # No category constraint, use the related term if available
                    # UPDATE USAGE TRACKING
                    self.update_term_usage(term, related_term)
                    return related_term
                
        # Default: return original term
        return term
    
    def find_valid_targets_in_image(self, image_id: int) -> List[Tuple[str, List[int]]]:
        """
        Find valid target objects in the given image.
        A valid target must have a bounding box and be part of at least one triplet.
        
        Args:
            image_id: The ID of the image to find targets in
            
        Returns:
            List of tuples (object_name, bounding_box)
        """
        try:
            if self.scene_graph_df is None:
                # Load data for this image if not already loaded
                self.scene_graph_df = self._load_scene_graph_chunk([image_id])
                
                if self.scene_graph_df.empty:
                    self.logger.warning(f"No scene graph data found for image ID {image_id}")
                    return []
                    
            # Filter scene graph for the specific image
            image_sg = self.scene_graph_df[self.scene_graph_df['image_id'] == image_id]
            
            if image_sg.empty:
                self.logger.warning(f"Image ID {image_id} not found in loaded scene graph data")
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
            
        except Exception as e:
            self.logger.error(f"Error finding valid targets for image {image_id}: {e}")
            return []
    
    def has_bounding_box(self, image_id: int, entity: str) -> bool:
        """
        Check if an entity has a bounding box in the given image.
        
        Args:
            image_id: The image ID
            entity: The entity name to check
            
        Returns:
            True if the entity has a bounding box, False otherwise
        """
        try:
            image_sg = self.scene_graph_df[self.scene_graph_df['image_id'] == image_id]
            
            # Check if entity appears as subject with bbox
            subject_match = image_sg[
                (image_sg['subject'] == entity) & 
                (image_sg['subject_bbox'].notna())
            ]
            
            # Check if entity appears as object with bbox
            object_match = image_sg[
                (image_sg['object'] == entity) & 
                (image_sg['object_bbox'].notna()) &
                (image_sg['object'] != 'NULL')
            ]
            
            return not subject_match.empty or not object_match.empty
            
        except Exception as e:
            self.logger.error(f"Error checking bounding box for entity {entity}: {e}")
            return False
    
    def find_triplets_for_target(self, image_id: int, target: str, target_bbox: List[int]) -> List[Dict[str, Any]]:
        """
        Find all triplets in the scene graph that involve the target.
        Uses caching for performance.
        
        Args:
            image_id: The image ID
            target: The target object name
            target_bbox: The target bounding box
            
        Returns:
            List of dictionaries representing triplets
        """
        # Check cache first
        cache_key = (image_id, target, tuple(target_bbox) if isinstance(target_bbox, list) else target_bbox)
        if cache_key in self.triplet_cache:
            return self.triplet_cache[cache_key]
        
        try:
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
                    'other_bbox': row['object_bbox'],
                    'subject': row['subject'],
                    'subject_bbox': row['subject_bbox'],
                    'object': row['object'],
                    'object_bbox': row['object_bbox']
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
                    'other_bbox': row['subject_bbox'],
                    'subject': row['subject'],
                    'subject_bbox': row['subject_bbox'],
                    'object': row['object'],
                    'object_bbox': row['object_bbox']
                })
            
            # Cache the result
            self.triplet_cache[cache_key] = triplets
            return triplets
            
        except Exception as e:
            self.logger.error(f"Error finding triplets for target {target} in image {image_id}: {e}")
            return []
    
    def find_triplets_containing_entity(self, image_id: int, entity: str) -> List[Dict[str, Any]]:
        """
        Find all triplets that contain the given entity (as subject or object).
        
        Args:
            image_id: The image ID
            entity: The entity name to find
            
        Returns:
            List of dictionaries representing triplets
        """
        try:
            # Validate entity
            if not isinstance(entity, str) or len(entity.strip()) == 0 or entity == 'NULL':
                return []
                
            image_sg = self.scene_graph_df[self.scene_graph_df['image_id'] == image_id]
            triplets = []
            
            # Find triplets where entity is the subject
            subject_triplets = image_sg[image_sg['subject'] == entity]
            for _, row in subject_triplets.iterrows():
                # Validate the triplet data
                if (isinstance(row['subject'], str) and isinstance(row['object'], str) and 
                    isinstance(row['relationship'], str)):
                    triplets.append({
                        'relation': row['relationship'],
                        'subject': row['subject'],
                        'subject_bbox': row['subject_bbox'],
                        'object': row['object'],
                        'object_bbox': row['object_bbox']
                    })
            
            # Find triplets where entity is the object (and not NULL)
            object_triplets = image_sg[
                (image_sg['object'] == entity) & 
                (image_sg['object'] != 'NULL')
            ]
            for _, row in object_triplets.iterrows():
                # Validate the triplet data
                if (isinstance(row['subject'], str) and isinstance(row['object'], str) and 
                    isinstance(row['relationship'], str)):
                    triplets.append({
                        'relation': row['relationship'],
                        'subject': row['subject'],
                        'subject_bbox': row['subject_bbox'],
                        'object': row['object'],
                        'object_bbox': row['object_bbox']
                    })
            
            return triplets
            
        except Exception as e:
            self.logger.error(f"Error finding triplets containing entity {entity}: {e}")
            return []
    
    def build_triplet_chain(self, image_id: int, initial_target: str, initial_bbox: List[int], 
                           max_chain_length: int) -> List[Dict[str, Any]]:
        """
        Build a chain of connected triplets starting from the initial target.
        
        Args:
            image_id: The image ID
            initial_target: The initial target object
            initial_bbox: The initial target's bounding box
            max_chain_length: Maximum number of triplets in the chain
            
        Returns:
            List of triplets forming a connected chain
        """
        try:
            # Start with triplets involving the initial target
            initial_triplets = self.find_triplets_for_target(image_id, initial_target, initial_bbox)
            
            if not initial_triplets:
                return []
            
            # Select the first triplet randomly and mark the target
            first_triplet = random.choice(initial_triplets)
            first_triplet['target_entity'] = initial_target
            first_triplet['target_bbox'] = initial_bbox
            
            chain = [first_triplet]
            used_triplets = {self._triplet_key(first_triplet)}
            
            # Build the chain
            for _ in range(max_chain_length - 1):
                last_triplet = chain[-1]
                
                # Get potential next entities (subject and object of last triplet that aren't the target)
                next_entities = []
                
                # Add subject if it has a bounding box, is not target, and is not an attribute
                if (last_triplet['subject_bbox'] is not None and 
                    last_triplet['subject'] != 'NULL' and
                    last_triplet['subject'] != initial_target and
                    isinstance(last_triplet['subject'], str) and  # Ensure it's a valid string
                    len(last_triplet['subject'].strip()) > 0 and  # Not empty
                    self.has_bounding_box(image_id, last_triplet['subject'])):
                    next_entities.append(last_triplet['subject'])
                
                # Add object if it has a bounding box, is not target, and is not an attribute  
                if (last_triplet['object_bbox'] is not None and 
                    last_triplet['object'] != 'NULL' and
                    last_triplet['object'] != initial_target and
                    isinstance(last_triplet['object'], str) and  # Ensure it's a valid string
                    len(last_triplet['object'].strip()) > 0 and  # Not empty
                    self.has_bounding_box(image_id, last_triplet['object'])):
                    next_entities.append(last_triplet['object'])
                
                if not next_entities:
                    break
                
                # Find triplets for the next entities
                candidate_triplets = []
                for entity in next_entities:
                    entity_triplets = self.find_triplets_containing_entity(image_id, entity)
                    for triplet in entity_triplets:
                        triplet_key = self._triplet_key(triplet)
                        if triplet_key not in used_triplets:
                            candidate_triplets.append(triplet)
                
                if not candidate_triplets:
                    break
                
                # Select next triplet randomly
                next_triplet = random.choice(candidate_triplets)
                
                # Mark target information for this triplet
                next_triplet['target_entity'] = initial_target
                next_triplet['target_bbox'] = initial_bbox
                
                chain.append(next_triplet)
                used_triplets.add(self._triplet_key(next_triplet))
            
            return chain
            
        except Exception as e:
            self.logger.error(f"Error building triplet chain: {e}")
            return []
    
    def _triplet_key(self, triplet: Dict[str, Any]) -> Tuple:
        """Create a unique key for a triplet to avoid duplicates."""
        try:
            subject = triplet.get('subject', '')
            relation = triplet.get('relation', '')
            obj = triplet.get('object', '')
            subject_bbox = triplet.get('subject_bbox')
            object_bbox = triplet.get('object_bbox')
            
            # Ensure all components are strings
            subject = str(subject) if subject is not None else ''
            relation = str(relation) if relation is not None else ''
            obj = str(obj) if obj is not None else ''
            
            return (
                subject,
                relation,
                obj,
                tuple(subject_bbox) if isinstance(subject_bbox, list) else subject_bbox,
                tuple(object_bbox) if isinstance(object_bbox, list) else object_bbox
            )
        except Exception as e:
            self.logger.error(f"Error creating triplet key: {e}")
            return (str(triplet), )  # Fallback key
    
    def generate_expression_predicate_from_chain_triplet(self, triplet: Dict[str, Any]) -> Optional[str]:
        """
        Generate an expression predicate from a triplet in a chain, using only X as a variable
        for the target entity, and constant names for all other entities.
        
        Args:
            triplet: Dictionary containing the full triplet information
            
        Returns:
            Expression predicate as a string, or None if invalid
        """
        try:
            # Apply appropriate term relationships based on context
            relation = self.potentially_mutate_term(triplet['relation'], "relation_predicate")
            
            # Get target information
            target_entity = triplet.get('target_entity')
            target_bbox = triplet.get('target_bbox')
            
            # Handle subject
            subject = triplet['subject']
            if not isinstance(subject, str) or len(subject.strip()) == 0:
                return None
                
            if subject == target_entity:
                subject_var = 'X'  # Target is always X
            else:
                # Use constant name with potential mutation
                subject_var = self.potentially_mutate_term(subject, "expression_predicate")
            
            # Handle object
            obj = triplet['object']
            if obj == 'NULL' or not isinstance(obj, str) or len(obj.strip()) == 0:
                return None
                
            if obj == target_entity:
                object_var = 'X'  # Target is always X
            else:
                # Use constant name with potential mutation
                object_var = self.potentially_mutate_term(obj, "expression_predicate")
            
            # Skip if object doesn't have bbox when it should (except for attributes)
            if (triplet['object_bbox'] is None and 
                triplet['relation'] != 'hasAttribute' and 
                obj != target_entity):
                return None
            
            return f"expression({relation}, {subject_var}, {object_var})"
            
        except Exception as e:
            self.logger.error(f"Error generating expression predicate from chain triplet: {e}")
            return None
    
    def generate_expression_predicate(self, triplet: Dict[str, Any]) -> str:
        """
        Generate an expression predicate from a triplet, with semantic-preserving mutations.
        
        Args:
            triplet: Dictionary containing relation, position, other, other_bbox
            
        Returns:
            Expression predicate as a string, or None if the triplet is invalid
        """
        # Apply appropriate term relationships based on context
        relation = self.potentially_mutate_term(triplet['relation'], "relation_predicate")
        other = triplet['other']
        
        # Skip if other is NULL or other doesn't have a bbox when needed
        if other == 'NULL' or (triplet['other_bbox'] is None and triplet['relation'] != 'hasAttribute'):
            return None
            
        # Potentially mutate the other term with expression predicate context
        other = self.potentially_mutate_term(other, "expression_predicate")
        
        if triplet['position'] == 'subject':
            # Target is subject, so expression is (relation, X, other)
            return f"expression({relation}, X, {other})"
        else:
            # Target is object, so expression is (relation, other, X)
            return f"expression({relation}, {other}, X)"
    
    def generate_negative_instance(self, image_id: int, query_text: str, original_target: Tuple[str, List[int]]) -> Optional[Dict[str, Any]]:
        """
        Generate a negative instance by selecting a different target for the same query.
        FIXED: Better target comparison to prevent same target being selected.
        
        Args:
            image_id: The image ID
            query_text: The original query text
            original_target: Tuple of (target_name, target_bbox)
            
        Returns:
            Negative query dictionary or None if no suitable negative target found
        """
        try:
            # Find all valid targets in the image
            valid_targets = self.find_valid_targets_in_image(image_id)
            
            # Remove the original target from candidates
            original_name, original_bbox = original_target
            
            # FIXED: Better comparison logic
            negative_candidates = []
            for name, bbox in valid_targets:
                # Convert bbox to list if it's not already
                if isinstance(bbox, tuple):
                    bbox = list(bbox)
                if isinstance(original_bbox, tuple):
                    original_bbox = list(original_bbox)
                    
                # Check if this is different from original target
                if not (name == original_name and bbox == original_bbox):
                    negative_candidates.append((name, bbox))
            
            if not negative_candidates:
                self.logger.warning(f"No negative candidates found for image {image_id}")
                return None
            
            # Select a random negative target
            negative_target_name, negative_target_bbox = random.choice(negative_candidates)
            
            # FIXED: Ensure target format consistency
            return {
                "image_id": image_id,
                "query": query_text,
                "target": [negative_target_name, negative_target_bbox],  # Keep consistent format
                "probability": 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error generating negative instance: {e}")
            return None
    
    def generate_diverse_queries(self, image_id: int, num_queries: int = 3, diversity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Generate diverse queries by ensuring they use different relations and targets.
        FIXED: Added triplet deduplication within queries.
        
        Args:
            image_id: The image ID
            num_queries: Number of queries to generate
            diversity_threshold: Threshold for allowing non-diverse queries
            
        Returns:
            List of query dictionaries
        """
        queries = []
        used_relations = set()
        used_targets = set()
        
        valid_targets = self.find_valid_targets_in_image(image_id)
        
        if not valid_targets:
            self.logger.warning(f"No valid targets found for image {image_id}")
            return []
            
        # Try to generate the requested number of queries
        attempts = 0
        max_attempts = num_queries * 5  # Allow more attempts than needed queries
        
        while len(queries) < num_queries and attempts < max_attempts:
            attempts += 1
            
            # Prioritize unused targets for diversity
            available_targets = [t for t in valid_targets if t[0] not in used_targets]
            if not available_targets and random.random() > diversity_threshold:
                available_targets = valid_targets
                
            if not available_targets:
                continue
                
            # Randomly select a target
            target, target_bbox = random.choice(available_targets)
            
            # Decide whether to use chaining or traditional approach
            use_chaining = random.random() < 0.7  # 70% chance to use chaining
            
            if use_chaining:
                # Use the new chaining approach
                num_expr_predicates = random.randint(self.min_expr_predicates, self.max_expr_predicates)
                chain = self.build_triplet_chain(image_id, target, target_bbox, num_expr_predicates)
                
                if not chain:
                    continue
                
                # Generate expression predicates from the chain
                expr_predicates = []
                for triplet in chain:
                    predicate = self.generate_expression_predicate_from_chain_triplet(triplet)
                    if predicate:
                        expr_predicates.append(predicate)
                        # Track used relations for diversity
                        used_relations.add(triplet['relation'])
                
            else:
                # Use traditional approach with FIXED deduplication
                triplets = self.find_triplets_for_target(image_id, target, target_bbox)
                
                # Skip if no valid triplets
                if not triplets:
                    continue
                    
                # Filter out already used relations to increase diversity
                diverse_triplets = [t for t in triplets if t['relation'] not in used_relations]
                
                if not diverse_triplets and random.random() > diversity_threshold:
                    # Sometimes allow non-diverse triplets
                    diverse_triplets = triplets
                    
                if not diverse_triplets:
                    continue
                    
                # Select number of expression predicates
                num_expr_predicates = random.randint(self.min_expr_predicates, 
                                                   min(self.max_expr_predicates, len(diverse_triplets)))
                
                # FIXED: Deduplicate triplets within single query
                selected_triplets = []
                used_triplet_keys = set()
                
                # Randomly select triplets without duplicates
                shuffled_triplets = diverse_triplets.copy()
                random.shuffle(shuffled_triplets)
                
                for triplet in shuffled_triplets:
                    if len(selected_triplets) >= num_expr_predicates:
                        break
                        
                    triplet_key = self._triplet_key(triplet)
                    if triplet_key not in used_triplet_keys:
                        selected_triplets.append(triplet)
                        used_triplet_keys.add(triplet_key)
                
                # Generate expression predicates
                expr_predicates = []
                for triplet in selected_triplets:
                    predicate = self.generate_expression_predicate(triplet)
                    if predicate:
                        expr_predicates.append(predicate)
                        # Track used relations for diversity
                        used_relations.add(triplet['relation'])
            
            # Skip if no valid expression predicates
            if not expr_predicates:
                continue
                
            # Get the hypernym for the target object for the type predicate
            # Always use hypernym for types (more general)
            target_type = self.get_hypernym(target)
            
            # Construct the query
            query_text = f"target(X), type(X, {target_type}), {', '.join(expr_predicates)}"
            
            # Generate positive instance
            positive_query = {
                "image_id": image_id,
                "query": query_text,
                "target": [target, target_bbox],
                "probability": 1.0
            }
            
            queries.append(positive_query)
            
            # Generate negative instance with specified probability
            if random.random() < self.negative_ratio:
                negative_query = self.generate_negative_instance(image_id, query_text, (target, target_bbox))
                if negative_query:
                    queries.append(negative_query)
            
            # Track used targets for diversity
            used_targets.add(target)
            
        if len(queries) < num_queries:
            self.logger.warning(f"Could only generate {len(queries)}/{num_queries} queries for image {image_id}")
            
        return queries
    
    def build_term_to_images_index(self) -> Dict[str, Set[int]]:
        """Build an index of which terms appear in which images."""
        self.logger.info("Building term-to-images index...")
        
        term_to_images = defaultdict(set)
        
        # Read the scene graph in chunks and build the index
        if self.is_json:
            with open(self.scene_graph_path, 'r') as f:
                data = json.load(f)
            for item in data:
                image_id = item.get('image_id')
                if image_id:
                    term_to_images[item.get('subject', '')].add(image_id)
                    term_to_images[item.get('object', '')].add(image_id)
        else:
            for chunk in pd.read_csv(self.scene_graph_path, chunksize=self.chunk_size):
                for _, row in chunk.iterrows():
                    image_id = row['image_id']
                    term_to_images[row['subject']].add(image_id)
                    if row['object'] != 'NULL':
                        term_to_images[row['object']].add(image_id)
        
        # Remove empty keys
        term_to_images = {k: v for k, v in term_to_images.items() if k and v}
        
        self.logger.info(f"Built index for {len(term_to_images)} terms")
        return dict(term_to_images)
    
    def generate_queries_for_underused_terms(self, max_images_per_term: int = 2) -> List[Dict[str, Any]]:
        """Fast generation for underused terms - try up to 2 images per term."""
        
        if not self.underused_terms:
            return []
        
        # Build index if not exists
        if not hasattr(self, 'term_to_images_index'):
            self.term_to_images_index = self.build_term_to_images_index()
        
        queries = []
        
        for term in list(self.underused_terms):
            if self.term_usage_count[term] >= self.min_term_usage:
                continue
            
            # Get images containing this term
            candidate_images = list(self.term_to_images_index.get(term, set()))
            
            if not candidate_images:
                continue
            
            # Take up to 2 images per term
            selected_images = candidate_images[:max_images_per_term]
            
            term_queries = []
            for image_id in selected_images:
                # Load the image data
                self.scene_graph_df = self._load_scene_graph_chunk([image_id])
                if self.scene_graph_df.empty:
                    continue
                
                # Force 100% mutation
                original_prob = self.ontology_mutation_prob
                self.ontology_mutation_prob = 0.0
                
                try:
                    image_queries = self.generate_query_for_image(image_id, 10)
                    
                    # Check if term appears in generated query
                    for query in image_queries:
                        if term in query['query']:
                            ## swap the term with a related term
                            ##need to simply extract a related term from the
                            #ontlogy, directly from the ontology, not using
                            #potentially_mutate_term
                            related_term = self.apply_term_relationship(term, "expression_predicate")
                            query['query'] = query['query'].replace(term, related_term)
                            term_queries.append(query)
                            break  # Only take one query per image
                            
                finally:
                    self.ontology_mutation_prob = original_prob
            
            # Add all successful queries for this term (up to 2)
            queries.extend(term_queries)
            
            if term_queries:
                self.logger.info(f"Generated {len(term_queries)} queries for underused term '{term}'")
        
        return queries
    
    def generate_query_for_image(self, image_id: int, num_queries: int = 1) -> List[Dict[str, Any]]:
        """
        Generate queries for a specific image.
        
        Args:
            image_id: The image ID
            num_queries: Number of queries to generate
            
        Returns:
            List of query dictionaries
        """
        # Use the diverse query generation by default
        return self.generate_diverse_queries(image_id, num_queries)
    
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
        try:
            if self.is_json:
                self.logger.info(f"Loading JSON data for image IDs: {image_ids if image_ids else 'all'}")
                
                # For very large files, consider using streaming approaches
                with open(self.scene_graph_path, 'r') as f:
                    # Memory-efficient approach for large files would use ijson here
                    # But for simplicity, we'll load it all
                    data = json.load(f)
                    
                # If specific image IDs are requested, filter the data
                if image_ids:
                    filtered_data = []
                    for item in data:
                        if item.get('image_id') in image_ids:
                            filtered_data.append(item)
                    chunk_df = pd.DataFrame(filtered_data)
                else:
                    # Load a chunk
                    chunk_df = pd.DataFrame(data[:self.chunk_size])
            else:
                self.logger.info(f"Loading CSV data for image IDs: {image_ids if image_ids else 'next chunk'}")
                
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
                self.logger.info(f"Processing bounding boxes for {len(chunk_df)} rows")
                chunk_df['subject_bbox'] = chunk_df['subject_bbox'].apply(
                    lambda x: ast.literal_eval(x) if x != 'NULL' and pd.notna(x) else None
                )
                chunk_df['object_bbox'] = chunk_df['object_bbox'].apply(
                    lambda x: ast.literal_eval(x) if x != 'NULL' and pd.notna(x) else None
                )
                
            return chunk_df
            
        except Exception as e:
            self.logger.error(f"Error loading scene graph chunk: {e}")
            return pd.DataFrame()
    
    def get_unique_image_ids(self, max_images=None):
        """
        Get unique image IDs from the scene graph.
        
        Args:
            max_images: Maximum number of image IDs to return
            
        Returns:
            List of unique image IDs
        """
        try:
            if self.is_json:
                self.logger.info("Getting unique image IDs from JSON file")
                # For JSON, we need to scan the file to get unique IDs
                with open(self.scene_graph_path, 'r') as f:
                    data = json.load(f)
                
                # Extract unique image IDs
                image_ids = set(item.get('image_id') for item in data if 'image_id' in item)
                image_ids = list(image_ids)
            else:
                self.logger.info("Getting unique image IDs from CSV file")
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
            
        except Exception as e:
            self.logger.error(f"Error getting unique image IDs: {e}")
            return []
        
    def _process_batch(self, batch_image_ids, queries_per_image):
        """
        Process a batch of image IDs and generate queries.
        Used for parallel processing.
        
        Args:
            batch_image_ids: List of image IDs to process
            queries_per_image: Tuple (min, max) of queries per image
            
        Returns:
            List of generated query dictionaries
        """
        batch_queries = []
        
        # Load the scene graph data for this batch
        self.scene_graph_df = self._load_scene_graph_chunk(batch_image_ids)
        
        # Skip if no data was found
        if self.scene_graph_df.empty:
            return batch_queries
            
        for image_id in batch_image_ids:
            # Check if this image ID exists in the loaded data
            if image_id not in self.scene_graph_df['image_id'].values:
                continue
                
            num_queries = random.randint(queries_per_image[0], queries_per_image[1])
            image_queries = self.generate_query_for_image(image_id, num_queries)
            batch_queries.extend(image_queries)
            
        return batch_queries
        
    def generate_queries_in_batches(self, output_path: str, batch_size: int = 100, 
                                   queries_per_image: Tuple[int, int] = (1, 3),
                                   resume: bool = False, save_interval: int = 10,
                                   max_images: Optional[int] = None,
                                   use_multiprocessing: bool = False,
                                   num_processes: int = 4,
                                   handle_underused_terms: bool = True):
        """
        Generate queries in batches and save incrementally to a JSON file.
        ENHANCED: Now includes handling of underused ontology terms.
        
        Args:
            output_path: Path to save the JSON file
            batch_size: Number of images to process in each batch
            queries_per_image: Tuple (min, max) of queries to generate per image
            resume: Whether to resume from an existing file
            save_interval: Save after processing this many images
            max_images: Maximum number of images to process (None for all)
            use_multiprocessing: Whether to use multiprocessing
            num_processes: Number of processes to use if multiprocessing
            handle_underused_terms: Whether to generate additional queries for underused terms
        """
        # Get unique image IDs, optionally limited to max_images
        self.logger.info("Getting unique image IDs...")
        image_ids = self.get_unique_image_ids(max_images)
        self.logger.info(f"Found {len(image_ids)} unique image IDs")
        
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
                        
                self.logger.info(f"Resuming from existing file with {len(existing_queries)} queries and {len(processed_image_ids)} processed images.")
            except Exception as e:
                self.logger.error(f"Error loading existing file: {e}. Starting fresh.")
                existing_queries = []
                processed_image_ids = set()
        
        # Filter out already processed image IDs
        remaining_image_ids = [img_id for img_id in image_ids if img_id not in processed_image_ids]
        self.logger.info(f"Remaining images to process: {len(remaining_image_ids)}")
        
        # Process in batches
        all_queries = existing_queries
        
        if use_multiprocessing:
            from multiprocessing import Pool
            
            # Split work into chunks for parallel processing
            num_processes = min(num_processes, os.cpu_count() or 4)
            chunks = [remaining_image_ids[i:i+batch_size] 
                     for i in range(0, len(remaining_image_ids), batch_size)]
            
            self.logger.info(f"Using multiprocessing with {num_processes} processes for {len(chunks)} batches")
            
            # Process batches in parallel
            with Pool(num_processes) as pool:
                results = pool.starmap(self._process_batch, [(chunk, queries_per_image) for chunk in chunks])
                
            # Combine results
            for batch_queries in results:
                all_queries.extend(batch_queries)
                
            # Save intermediate results
            self.save_queries(output_path, all_queries)
            
        else:
            # Sequential processing
            processed_count = 0
            
            for i in range(0, len(remaining_image_ids), batch_size):
                batch_image_ids = remaining_image_ids[i:i+batch_size]
                batch_queries = []
                
                self.logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_image_ids)} images...")
                
                # Load the scene graph data for this batch of image IDs
                self.logger.info("Loading scene graph data for this batch...")
                self.scene_graph_df = self._load_scene_graph_chunk(batch_image_ids)
                
                # Skip if no data was found for these image IDs
                if self.scene_graph_df.empty:
                    self.logger.warning(f"No scene graph data found for batch {i//batch_size + 1}. Skipping.")
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
                
                self.logger.info(f"Batch completed. Total queries so far: {len(all_queries)}")
                
                # Clear the dataframe to free memory
                self.scene_graph_df = None
        
        # ENHANCED: Handle underused terms
        if handle_underused_terms:
            self.logger.info("Checking for underused ontology terms...")
            underused_queries = self.generate_queries_for_underused_terms()
            
            if underused_queries:
                all_queries.extend(underused_queries)
                self.logger.info(f"Added {len(underused_queries)} queries for underused terms")
        
        # Final save
        self.save_queries(output_path, all_queries)
        
        # Save term usage report
        usage_report_path = output_path.replace('.json', '_term_usage_report.json')
        self.save_term_usage_report(usage_report_path)
        
        # Log statistics about positive vs negative instances
        positive_count = sum(1 for q in all_queries if q['probability'] == 1.0)
        negative_count = sum(1 for q in all_queries if q['probability'] == 0.0)
        
        self.logger.info(f"All done! Generated a total of {len(all_queries)} queries:")
        self.logger.info(f"- Positive instances: {positive_count}")
        self.logger.info(f"- Negative instances: {negative_count}")
        self.logger.info(f"- Negative ratio: {negative_count / len(all_queries) * 100:.1f}%")
        self.logger.info(f"Saved to {output_path}")
    
    def save_queries(self, output_path: str, queries: List[Dict[str, Any]]):
        """
        Save queries to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
            queries: List of query dictionaries
        """
        try:
            # Convert NumPy types to native Python types
            queries_copy = self.convert_numpy_types(queries)
            
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Wrap in the expected format
            queries_data = {"queries": queries_copy}
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(queries_data, f, indent=4)
            
            self.logger.info(f"Saved {len(queries)} queries to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving queries to {output_path}: {e}")
    
    def update_existing_queries(self, existing_path: str, output_path: str, 
                                new_image_ids: List[int], queries_per_image: Tuple[int, int] = (1, 3)):
        """
        Generate queries for additional images and merge with existing dataset.
        
        Args:
            existing_path: Path to existing queries JSON file
            output_path: Path to save the updated JSON file
            new_image_ids: List of new image IDs to process
            queries_per_image: Tuple (min, max) of queries per image
            
        Returns:
            Number of new queries generated
        """
        try:
            # Load existing queries
            with open(existing_path, 'r') as f:
                existing_data = json.load(f)
                existing_queries = existing_data.get('queries', [])
            
            # Get IDs that are already processed
            processed_ids = {q['image_id'] for q in existing_queries}
            
            # Only process new IDs
            image_ids_to_process = [id for id in new_image_ids if id not in processed_ids]
            
            self.logger.info(f"Found {len(existing_queries)} existing queries")
            self.logger.info(f"Processing {len(image_ids_to_process)} new image IDs")
            
            # Generate new queries
            new_queries = []
            for image_id in tqdm(image_ids_to_process):
                num_queries = random.randint(queries_per_image[0], queries_per_image[1])
                image_queries = self.generate_query_for_image(image_id, num_queries)
                new_queries.extend(image_queries)
            
            # Merge and save
            all_queries = existing_queries + new_queries
            self.save_queries(output_path, all_queries)
            
            self.logger.info(f"Updated queries with {len(new_queries)} new queries")
            return len(new_queries)
            
        except Exception as e:
            self.logger.error(f"Error updating queries: {e}")
            return 0


if __name__ == "__main__":
    # Paths to your data (override via config/CLI as needed)
    scene_graph_path = "data/vocab_filtered_vg_scene_graph.csv"
    ontology_path = "data/final_ontology.csv"
    output_path = "data/query/generated_queries.json"
    
    # Initialize the query generator with enhanced tracking
    generator = QueryGenerator(
        scene_graph_path=scene_graph_path,
        ontology_path=ontology_path,
        ontology_mutation_prob=0.5,
        min_expr_predicates=1,
        max_expr_predicates=4,
        chunk_size=100,  # Adjust based on your memory constraints
        negative_ratio=0.4,  # 30% of queries will be negative instances
        min_term_usage=5  # Each ontology term should be used at least 10 times
    )
    
    # Use the enhanced batch processing method
    generator.generate_queries_in_batches(
        output_path=output_path,
        batch_size=50,            # Process 50 images at a time
        queries_per_image=(1, 3), # Generate 1-3 queries per image
        resume=True,              # Resume from existing file if it exists
        save_interval=10,         # Save after processing 10 images
        max_images=3000,          # Only process the first 1000 images
        use_multiprocessing=True, # Enable multiprocessing for faster generation
        num_processes=4,          # Use 4 processes
        handle_underused_terms=True  # Generate additional queries for underused terms
    )