import csv
import math
import re
import time
import torch
import numpy as np
from typing import List, Dict, Any
from collections import Counter
from data.dataset import DatasetInstance
from train.trainer import ReferringTrainer
from deepsoftlog.data import query_to_prolog
from deepsoftlog.data.query import Query


class EnhancedReferringEvaluator(ReferringTrainer):
    def __init__(self, program, config, **search_args):
        # Initialize without optimizer since we're not training
        self.program = program
        self.config = config
        self.search_args = search_args
        self.vocab_coverage = {
            'new_constants': set(),
            'total_new': 0,
            'instances_with_new_vocab': 0
        }
        
        # Track initial vocabulary
        self.initial_vocab = set(program.store.constant_embeddings.keys())
        
    def evaluate(self, eval_dataloader) -> List[Dict[str, Any]]:
        """Main evaluation loop"""
        self.program.store.eval()  # Set to evaluation mode
        all_results = []
        
        for batch_idx, instances in enumerate(eval_dataloader):
            print(f"Evaluating batch {batch_idx + 1}/{len(eval_dataloader)}")
            
            for instance in instances:
                result = self.evaluate_single_instance(instance, batch_idx)
                all_results.append(result)
        
        return all_results
    
    def evaluate_single_instance(self, instance: DatasetInstance, batch_idx: int) -> Dict[str, Any]:
        """Evaluate a single instance with comprehensive analysis"""
        print(f"Evaluating instance with query: {instance.query}")
        
        # Start timing
        start_time = time.time()
        
        # Track vocabulary before update
        vocab_before = set(self.program.store.constant_embeddings.keys())
        
        # Update program clauses with scene graph
        self.program.update_clauses(instance)
        
        # Track new vocabulary after update
        vocab_after = set(self.program.store.constant_embeddings.keys())
        new_vocab = vocab_after - vocab_before
        self._update_vocab_coverage(new_vocab)
        
        # Analyze scene graph
        scene_graph_analysis = self._analyze_scene_graph(instance)
        
        # Prepare query
        if not isinstance(instance.query, Query):
            instance.query = query_to_prolog(instance.query, p=instance.metadata.get('probability', 1.0))
        
        # Analyze query complexity
        query_analysis = self._analyze_query_complexity(instance.query)
        
        # Get results from program with detailed tracking
        with torch.no_grad():
            results_dict, proof_analysis = self._get_detailed_results(instance.query)
        
        print(f"Raw results: {results_dict}")
        
        # Calculate timing
        evaluation_time = time.time() - start_time
        
        # Extract detailed results for all proven queries
        detailed_results = []
        for query_result, log_prob in results_dict.items():
            detailed_result = self._extract_result_details(
                query_result, log_prob, instance, batch_idx, new_vocab, 
                scene_graph_analysis, query_analysis, proof_analysis
            )
            detailed_results.append(detailed_result)
        
        # Calculate confidence metrics
        confidence_analysis = self._analyze_confidence(detailed_results)
        
        # Handle no results case
        if not detailed_results:
            detailed_results.append(self._create_enhanced_no_result_entry(
                instance, batch_idx, new_vocab, scene_graph_analysis, 
                query_analysis, proof_analysis, confidence_analysis
            ))
        
        # Find best result (highest probability)
        best_result = max(detailed_results, key=lambda x: x['probability'])
        
        # Create comprehensive summary result
        summary_result = {
            'image_id': instance.metadata.get('image_id', 'unknown'),
            'query': str(instance.query.query),
            'target_probability': instance.query.p,
            'batch_idx': batch_idx,
            'evaluation_time': evaluation_time,
            
            # Vocabulary analysis
            'new_vocab_count': len(new_vocab),
            'new_vocab_items': list(new_vocab),
            'vocab_coverage_ratio': len(vocab_after) / len(self.initial_vocab),
            
            # Results summary
            'total_results': len(detailed_results),
            'best_predicted_object': best_result['predicted_object'],
            'best_predicted_bbox': best_result['predicted_bbox'],
            'best_probability': best_result['probability'],
            'best_loss': best_result['loss'],
            
            # Scene graph analysis
            **scene_graph_analysis,
            
            # Query analysis
            **query_analysis,
            
            # Proof analysis
            **proof_analysis,
            
            # Confidence analysis
            **confidence_analysis,
            
            'all_results': detailed_results
        }
        
        # Add ground truth if available
        if hasattr(instance, 'target') and instance.target:
            summary_result['ground_truth_object'] = instance.target[0]
            summary_result['ground_truth_bbox'] = instance.target[1]
            summary_result['correct_prediction'] = (
                best_result['predicted_object'] == instance.target[0]
            )
        
        return summary_result
    
    def _get_detailed_results(self, query):
        """Get results with detailed proof tracking"""
        # Track proof attempts
        original_query_method = self.program.query
        proof_attempts = []
        search_stats = {
            'max_depth_reached': 0,
            'total_proof_attempts': 0,
            'search_limits_hit': [],
            'partial_matches': 0
        }
        
        def tracking_query(*args, **kwargs):
            # This would need integration with the proof system to track attempts
            result = original_query_method(*args, **kwargs)
            search_stats['total_proof_attempts'] += 1
            return result
        
        # Temporarily replace query method
        self.program.query = tracking_query
        
        try:
            results_dict = self.program.query(query.query, **self.search_args)
        finally:
            # Restore original method
            self.program.query = original_query_method
        
        return results_dict, search_stats
    
    def _analyze_scene_graph(self, instance):
        """Analyze scene graph coverage and utilization"""
        scene_graph = instance.scene_graph
        
        total_facts = len(scene_graph.triplets)
        total_objects = len(scene_graph.bounding_boxes)
        total_attributes = len(getattr(scene_graph, 'attributes', {}))
        
        # Analyze relationship types
        relationships = [triplet[1] for triplet in scene_graph.triplets]
        relationship_counts = Counter(relationships)
        
        # Analyze object types
        object_types = [obj_info[0] for obj_info in scene_graph.bounding_boxes.values()]
        object_type_counts = Counter(object_types)
        
        return {
            'scene_graph_facts_total': total_facts,
            'scene_graph_objects_total': total_objects,
            'scene_graph_attributes_total': total_attributes,
            'relationship_types': list(relationship_counts.keys()),
            'most_common_relationship': relationship_counts.most_common(1)[0] if relationships else None,
            'object_types': list(object_type_counts.keys()),
            'most_common_object': object_type_counts.most_common(1)[0] if object_types else None,
            'scene_graph_diversity': len(relationship_counts) + len(object_type_counts)
        }
    
    def _analyze_query_complexity(self, query):
        """Analyze query complexity and structure"""
        query_str = str(query.query)
        
        # Count different predicate types
        type_count = query_str.count('type(')
        expression_count = query_str.count('expression(')
        target_count = query_str.count('target(')
        
        # Extract predicates from expressions
        expression_matches = re.findall(r'expression\(([^,]+),', query_str)
        spatial_predicates = ['on', 'in', 'under', 'above', 'nextTo', 'behind', 'inFrontOf', 'leftOf', 'rightOf', 'near']
        attribute_predicates = ['hasAttribute', 'hasProperty', 'wearing', 'holding']
        
        spatial_count = sum(1 for pred in expression_matches if pred.strip() in spatial_predicates)
        attribute_count = sum(1 for pred in expression_matches if pred.strip() in attribute_predicates)
        
        # Calculate complexity score
        complexity_score = (type_count * 0.2 + expression_count * 0.5 + 
                          spatial_count * 0.3 + attribute_count * 0.2) / 10.0
        
        return {
            'query_conjuncts': type_count + expression_count,
            'query_type_predicates': type_count,
            'query_expression_predicates': expression_count,
            'query_spatial_predicates': spatial_count,
            'query_attribute_predicates': attribute_count,
            'query_complexity_score': min(complexity_score, 1.0),
            'query_predicates_used': list(set(expression_matches))
        }
    
    def _analyze_confidence(self, detailed_results):
        """Analyze prediction confidence and uncertainty"""
        if not detailed_results or all(r['probability'] == 0 for r in detailed_results):
            return {
                'result_entropy': float('inf'),
                'top_2_margin': 0.0,
                'prediction_confidence': 'none',
                'alternative_objects': []
            }
        
        # Sort by probability
        sorted_results = sorted(detailed_results, key=lambda x: x['probability'], reverse=True)
        probs = [r['probability'] for r in sorted_results if r['probability'] > 0]
        
        # Calculate entropy
        if probs:
            total_prob = sum(probs)
            normalized_probs = [p/total_prob for p in probs]
            entropy = -sum(p * math.log(p) for p in normalized_probs if p > 0)
        else:
            entropy = 0
        
        # Calculate margin between top 2
        top_2_margin = 0.0
        if len(sorted_results) >= 2:
            top_2_margin = sorted_results[0]['probability'] - sorted_results[1]['probability']
        
        # Determine confidence level
        if not probs:
            confidence = 'none'
        elif sorted_results[0]['probability'] > 0.8 and top_2_margin > 0.3:
            confidence = 'high'
        elif sorted_results[0]['probability'] > 0.5 and top_2_margin > 0.1:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Get alternative objects
        alternatives = [
            {'object': r['predicted_object'], 'probability': r['probability']}
            for r in sorted_results[1:6]  # Top 5 alternatives
            if r['probability'] > 0
        ]
        
        return {
            'result_entropy': entropy,
            'top_2_margin': top_2_margin,
            'prediction_confidence': confidence,
            'alternative_objects': alternatives,
            'total_candidates': len([r for r in detailed_results if r['probability'] > 0])
        }
    
    def _extract_result_details(self, query_result, log_prob, instance, batch_idx, new_vocab, 
                               scene_graph_analysis, query_analysis, proof_analysis):
        """Extract detailed information from a single query result"""
        # Convert log probability to regular probability
        probability = math.exp(log_prob) if isinstance(log_prob, (int, float)) else math.exp(log_prob.item())
        
        # Calculate loss (negative log likelihood)
        loss = -log_prob if isinstance(log_prob, (int, float)) else -log_prob.item()
        
        # Extract object and bbox information
        predicted_object, predicted_bbox = self._extract_object_and_bbox(query_result, instance)
        
        # Analyze soft unifications used (this would need deeper integration)
        soft_unification_analysis = self._analyze_soft_unifications(query_result, instance)
        
        return {
            'image_id': instance.metadata.get('image_id', 'unknown'),
            'query': str(instance.query.query),
            'result_query': str(query_result),
            'predicted_object': predicted_object,
            'predicted_bbox': predicted_bbox,
            'log_probability': log_prob if isinstance(log_prob, (int, float)) else log_prob.item(),
            'probability': probability,
            'loss': loss,
            'batch_idx': batch_idx,
            'new_vocab_count': len(new_vocab),
            'new_vocab_items': list(new_vocab),
            **soft_unification_analysis
        }
    
    def _analyze_soft_unifications(self, query_result, instance):
        """Analyze soft unifications used in this result"""
        # This would need deeper integration with the proof system
        # For now, provide placeholder structure
        return {
            'soft_unifications_detected': 0,
            'avg_soft_unification_score': 0.0,
            'critical_soft_mappings': [],
            'embedding_similarities_used': []
        }
    
    def _create_enhanced_no_result_entry(self, instance, batch_idx, new_vocab, 
                                       scene_graph_analysis, query_analysis, 
                                       proof_analysis, confidence_analysis):
        """Create enhanced entry when no results are found"""
        # Determine failure mode
        failure_mode = self._determine_failure_mode(instance, scene_graph_analysis, query_analysis)
        
        return {
            'image_id': instance.metadata.get('image_id', 'unknown'),
            'query': str(instance.query.query),
            'result_query': 'NO_RESULT',
            'predicted_object': 'NONE',
            'predicted_bbox': None,
            'log_probability': float('-inf'),
            'probability': 0.0,
            'loss': float('inf'),
            'batch_idx': batch_idx,
            'new_vocab_count': len(new_vocab),
            'new_vocab_items': list(new_vocab),
            'failure_mode': failure_mode,
            'soft_unifications_detected': 0,
            'avg_soft_unification_score': 0.0,
            'critical_soft_mappings': [],
            'embedding_similarities_used': []
        }
    
    def _determine_failure_mode(self, instance, scene_graph_analysis, query_analysis):
        """Determine why the proof failed"""
        # Simple heuristics - could be enhanced with deeper proof tracking
        if scene_graph_analysis['scene_graph_facts_total'] == 0:
            return 'empty_scene_graph'
        elif query_analysis['query_complexity_score'] > 0.8:
            return 'query_too_complex'
        elif len(instance.query.query.arguments) > 10:  # Very rough heuristic
            return 'proof_search_timeout'
        else:
            return 'no_matching_facts'
    
    def _extract_object_and_bbox(self, query_result, instance):
        """Extract object name and bounding box from query result"""
        # Convert query result to string for parsing
        query_str = str(query_result)
        print(f"Parsing query result: {query_str}")
        
        # Method 1: Look for target(X) where X has been bound to an object
        target_match = re.search(r'target\(([^)]+)\)', query_str)
        
        if target_match:
            bound_object = target_match.group(1)
            # Remove soft term notation if present
            if bound_object.startswith('~'):
                bound_object = bound_object[1:]
            
            print(f"Found bound object from target(): {bound_object}")
            
            # Find corresponding bbox in scene graph
            predicted_bbox = self._find_bbox_for_object(bound_object, instance)
            return bound_object, predicted_bbox
        
        # Method 2: Look for object(~object_name, bbox_id) patterns in the result
        object_matches = re.findall(r'object\(~?([^,]+),\s*([^)]+)\)', query_str)
        if object_matches:
            # Take the first match
            obj_name, bbox_id = object_matches[0]
            print(f"Found object from object() pattern: {obj_name}, bbox_id: {bbox_id}")
            
            bbox_coords = instance.scene_graph.bounding_boxes.get(bbox_id.strip())
            if bbox_coords:
                return obj_name, bbox_coords[1]  # bbox_coords is (object_name, coordinates)
            return obj_name, None
        
        # Method 3: Extract object from any soft term in the query
        soft_terms = re.findall(r'~([a-zA-Z_][a-zA-Z0-9_]*)', query_str)
        if soft_terms:
            # Look for the object that appears in our scene graph
            for term in soft_terms:
                bbox = self._find_bbox_for_object(term, instance)
                if bbox is not None:
                    print(f"Found object from soft term: {term}")
                    return term, bbox
        
        print(f"Could not extract object from query result: {query_str}")
        return "unknown", None
    
    def _find_bbox_for_object(self, object_name, instance):
        """Find bounding box coordinates for a given object name"""
        # Search through bounding boxes to find matching object
        for bbox_id, (obj_name, bbox_coords) in instance.scene_graph.bounding_boxes.items():
            if obj_name == object_name:
                return bbox_coords
        
        # If not found, might be an attribute - check attributes
        if hasattr(instance.scene_graph, 'attributes') and instance.scene_graph.attributes:
            for attr_id, attr_name in instance.scene_graph.attributes.items():
                if attr_name == object_name:
                    # This is an attribute, not a physical object with bbox
                    return "attribute"
        
        # Try partial matching (in case of slight differences)
        for bbox_id, (obj_name, bbox_coords) in instance.scene_graph.bounding_boxes.items():
            if object_name.lower() in obj_name.lower() or obj_name.lower() in object_name.lower():
                print(f"Found partial match: {object_name} -> {obj_name}")
                return bbox_coords
        
        print(f"No bounding box found for object: {object_name}")
        print(f"Available objects: {[obj_name for _, (obj_name, _) in instance.scene_graph.bounding_boxes.items()]}")
        return None
    
    def _update_vocab_coverage(self, new_vocab):
        """Update vocabulary coverage statistics"""
        if new_vocab:
            self.vocab_coverage['new_constants'].update(new_vocab)
            self.vocab_coverage['total_new'] += len(new_vocab)
            self.vocab_coverage['instances_with_new_vocab'] += 1
    
    def save_results_to_csv(self, results: List[Dict[str, Any]], output_path: str):
        """Save enhanced results to CSV file"""
        print(f"Saving enhanced results to: {output_path}")
        
        # Flatten results - create one row per individual result
        flattened_results = []
        for summary in results:
            if summary['all_results']:
                for detail in summary['all_results']:
                    # Add summary info to each detailed result
                    row = detail.copy()
                    row.update({
                        'target_probability': summary['target_probability'],
                        'total_results_for_query': summary['total_results'],
                        'evaluation_time': summary['evaluation_time'],
                        'vocab_coverage_ratio': summary['vocab_coverage_ratio'],
                        'scene_graph_facts_total': summary['scene_graph_facts_total'],
                        'scene_graph_objects_total': summary['scene_graph_objects_total'],
                        'scene_graph_diversity': summary['scene_graph_diversity'],
                        'query_complexity_score': summary['query_complexity_score'],
                        'query_conjuncts': summary['query_conjuncts'],
                        'result_entropy': summary['result_entropy'],
                        'prediction_confidence': summary['prediction_confidence'],
                        'top_2_margin': summary['top_2_margin'],
                        'total_candidates': summary['total_candidates'],
                        'is_best_result': (detail == summary['all_results'][0]),
                    })
                    
                    # Add ground truth info if available
                    if 'ground_truth_object' in summary:
                        row['ground_truth_object'] = summary['ground_truth_object']
                        row['ground_truth_bbox'] = summary['ground_truth_bbox']
                        row['correct_prediction'] = (detail['predicted_object'] == summary['ground_truth_object'])
                    
                    flattened_results.append(row)
            else:
                # No results case
                flattened_results.append(summary)
        
        # Define comprehensive CSV columns
        columns = [
            # Basic info
            'image_id', 'batch_idx', 'query', 'result_query',
            'predicted_object', 'predicted_bbox', 'target_probability',
            'log_probability', 'probability', 'loss',
            
            # Timing and performance
            'evaluation_time',
            
            # Vocabulary analysis
            'new_vocab_count', 'new_vocab_items', 'vocab_coverage_ratio',
            
            # Scene graph analysis
            'scene_graph_facts_total', 'scene_graph_objects_total', 'scene_graph_diversity',
            
            # Query analysis
            'query_complexity_score', 'query_conjuncts',
            
            # Confidence analysis
            'result_entropy', 'prediction_confidence', 'top_2_margin', 'total_candidates',
            
            # Soft unification analysis
            'soft_unifications_detected', 'avg_soft_unification_score',
            
            # Results metadata
            'total_results_for_query', 'is_best_result'
        ]
        
        # Add ground truth columns if available
        if any('ground_truth_object' in r for r in flattened_results):
            columns.extend(['ground_truth_object', 'ground_truth_bbox', 'correct_prediction'])
        
        # Add failure analysis for failed cases
        if any('failure_mode' in r for r in flattened_results):
            columns.append('failure_mode')
        
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for row in flattened_results:
                # Handle list fields
                for list_field in ['new_vocab_items', 'critical_soft_mappings', 'embedding_similarities_used']:
                    if list_field in row and isinstance(row[list_field], list):
                        row[list_field] = ';'.join(str(x) for x in row[list_field])
                
                # Handle bbox coordinates
                if 'predicted_bbox' in row and isinstance(row['predicted_bbox'], list):
                    row['predicted_bbox'] = str(row['predicted_bbox'])
                
                if 'ground_truth_bbox' in row and isinstance(row['ground_truth_bbox'], list):
                    row['ground_truth_bbox'] = str(row['ground_truth_bbox'])
                
                # Only write columns that exist in the row
                filtered_row = {k: row.get(k, '') for k in columns}
                writer.writerow(filtered_row)
        
        # Print enhanced summary statistics
        self._print_enhanced_evaluation_summary(results, flattened_results)
    
    def _print_enhanced_evaluation_summary(self, summaries, detailed_results):
        """Print enhanced evaluation summary statistics"""
        print("\n" + "="*60)
        print("ENHANCED EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Total instances: {len(summaries)}")
        print(f"Total individual results: {len(detailed_results)}")
        
        # Timing analysis
        times = [s.get('evaluation_time', 0) for s in summaries]
        print(f"\nTiming Analysis:")
        print(f"  Average time per instance: {np.mean(times):.3f}s")
        print(f"  Total evaluation time: {sum(times):.1f}s")
        
        # Vocabulary coverage
        print(f"\nVocabulary Coverage:")
        print(f"  New constants encountered: {len(self.vocab_coverage['new_constants'])}")
        print(f"  Total new vocab instances: {self.vocab_coverage['total_new']}")
        print(f"  Instances with new vocab: {self.vocab_coverage['instances_with_new_vocab']}")
        
        # Scene graph analysis
        sg_facts = [s.get('scene_graph_facts_total', 0) for s in summaries]
        sg_diversity = [s.get('scene_graph_diversity', 0) for s in summaries]
        print(f"\nScene Graph Analysis:")
        print(f"  Average facts per scene: {np.mean(sg_facts):.1f}")
        print(f"  Average diversity score: {np.mean(sg_diversity):.1f}")
        
        # Query complexity
        complexity = [s.get('query_complexity_score', 0) for s in summaries]
        print(f"\nQuery Complexity:")
        print(f"  Average complexity score: {np.mean(complexity):.3f}")
        
        # Confidence analysis
        confidence_levels = [r.get('prediction_confidence', 'none') for r in detailed_results]
        confidence_counts = Counter(confidence_levels)
        print(f"\nConfidence Distribution:")
        for level, count in confidence_counts.items():
            print(f"  {level}: {count} ({count/len(detailed_results)*100:.1f}%)")
        
        # Failure analysis
        failure_modes = [r.get('failure_mode') for r in detailed_results if r.get('failure_mode')]
        if failure_modes:
            failure_counts = Counter(failure_modes)
            print(f"\nFailure Mode Analysis:")
            for mode, count in failure_counts.items():
                print(f"  {mode}: {count}")
        
        # Results statistics
        results_with_answers = [r for r in detailed_results if r['predicted_object'] != 'NONE']
        print(f"\nResults Statistics:")
        print(f"  Instances with results: {len(results_with_answers)}")
        print(f"  Instances with no results: {len(detailed_results) - len(results_with_answers)}")
        
        if results_with_answers:
            probs = [r['probability'] for r in results_with_answers]
            entropies = [r.get('result_entropy', 0) for r in results_with_answers if r.get('result_entropy', 0) != float('inf')]
            print(f"  Average probability: {np.mean(probs):.4f}")
            print(f"  Average entropy: {np.mean(entropies):.4f}")
        
        # Accuracy if ground truth available
        if any('correct_prediction' in r for r in detailed_results):
            correct = sum(1 for r in detailed_results if r.get('correct_prediction', False))
            total = len([r for r in detailed_results if 'correct_prediction' in r])
            accuracy = correct / total if total > 0 else 0
            print(f"\nAccuracy: {accuracy:.3f} ({correct}/{total})")
        
        print("="*60)