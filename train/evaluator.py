import csv
import math
import re
import torch
import numpy as np
from typing import List, Dict, Any
from data.dataset import DatasetInstance
from train.trainer import ReferringTrainer 
from deepsoftlog.data import query_to_prolog
from deepsoftlog.data.query import Query
from train.train import fix_pretrained_checkpoint



class ReferringEvaluator(ReferringTrainer):
    def __init__(self, program, config, **search_args):
        print(f"🔍 ReferringEvaluator.__init__: input program store has {len(program.store.constant_embeddings)} embeddings")
        # Initialize without optimizer since we're not training
        self.program = program
        self.config = config
        self.search_args = search_args
        self.vocab_coverage = {
            'new_constants': set(),
            'total_new': 0,
            'instances_with_new_vocab': 0
        }
        print(f"🔍 ReferringEvaluator.__init__: after assignment, self.program store has {len(self.program.store.constant_embeddings)} embeddings")
        
        # Track initial vocabulary
        self.initial_vocab = set(program.store.constant_embeddings.keys())
        
    def evaluate(self, eval_dataloader) -> List[Dict[str, Any]]:
        """Main evaluation loop"""
        self.program.store.eval()  # Set to evaluation mode
        all_results = []
        
        for batch_idx, instances in enumerate(eval_dataloader):
            print(f"Evaluating batch {batch_idx + 1}/{len(eval_dataloader)}")
            
            for instance in instances:
                try:
                    result = self.evaluate_single_instance(instance, batch_idx)
                    all_results.append(result)
                except Exception as e:
                    print(f"❌ Error evaluating instance in batch {batch_idx + 1}: {e}")
                    print(f"   Query: {getattr(instance, 'query', 'Unknown')}")
                    print(f"   Target: {getattr(instance, 'target', 'Unknown')}")
                    print(f"   Continuing with next instance...")
                    
                    # Optionally, create a placeholder result for failed instances
                    error_result = {
                        'image_id': getattr(instance, 'metadata', {}).get('image_id', 'unknown'),
                        'query': str(getattr(instance, 'query', 'Unknown')),
                        'batch_idx': batch_idx,
                        'error': str(e),
                        'best_predicted_object': 'ERROR',
                        'best_predicted_bbox': None,
                        'best_probability': 0.0,
                        'best_loss': float('inf'),
                        'all_results': []
                    }
                    all_results.append(error_result)
        
        return all_results

    
    def evaluate_single_instance(self, instance: DatasetInstance, batch_idx: int) -> Dict[str, Any]:
        """Evaluate a single instance and extract detailed results"""
        print(f"Evaluating instance with query: {instance.query}")

        # Track vocabulary before update
        vocab_before = set(self.program.store.constant_embeddings.keys())
        
        # Update program clauses with scene graph
        self.program.update_clauses(instance)
        
        # Track new vocabulary after update
        vocab_after = set(self.program.store.constant_embeddings.keys())
        new_vocab = vocab_after - vocab_before
        self._update_vocab_coverage(new_vocab)
        
        # Prepare query
        if not isinstance(instance.query, Query):
            instance.query = query_to_prolog(instance.query, p=instance.metadata.get('probability', 1.0))
        
        # Get results from program
        with torch.no_grad():
            results_dict = self.program.query(instance.query.query, **self.search_args)
        
        print(f"Raw results: {results_dict}")
        print(f"Number of results found: {len(results_dict)}")  # DEBUG: Show how many results
        
        # Extract detailed results for all proven queries
        detailed_results = []
        for query_result, log_prob in results_dict.items():
            detailed_result = self._extract_result_details(
                query_result, log_prob, instance, batch_idx, new_vocab
            )
            detailed_results.append(detailed_result)
        
        # If no results, create a default entry
        if not detailed_results:
            detailed_results.append(self._create_no_result_entry(instance, batch_idx, new_vocab))
        
        # CRITICAL FIX: Sort results by probability (highest first)
        detailed_results.sort(key=lambda x: x.get('probability', 0), reverse=True)
        
        print(f"After sorting, detailed_results has {len(detailed_results)} results:")  # DEBUG
        for i, result in enumerate(detailed_results):
            print(f"  Rank {i+1}: {result['predicted_object']} (prob: {result['probability']:.6f})")
        
        # Find best result (now guaranteed to be first after sorting)
        best_result = detailed_results[0]
        
        # Create summary result
        summary_result = {
            'image_id': instance.metadata.get('image_id', 'unknown'),
            'batch_idx': batch_idx,
            'query': str(instance.query.query),
            'target_probability': instance.query.p,
            'new_vocab_count': len(new_vocab),
            'new_vocab_items': list(new_vocab),
            'total_results': len(detailed_results),
            'best_predicted_object': best_result['predicted_object'],
            'best_predicted_bbox': best_result['predicted_bbox'],
            'best_probability': best_result['probability'],
            'best_loss': best_result['loss'],
            'all_results': detailed_results  # Now properly sorted
        }
        
        # Add ground truth if available
        if hasattr(instance, 'target') and instance.target:
            summary_result['ground_truth_object'] = instance.target[0]
            summary_result['ground_truth_bbox'] = instance.target[1]
            summary_result['correct_prediction'] = (
                best_result['predicted_object'] == instance.target[0]
            )
        
        return summary_result
    


    
   
    
    def _create_no_result_entry(self, instance, batch_idx, new_vocab):
        """Create entry when no results are found"""
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
        }
    
    
    def _extract_result_details(self, query_result, log_prob, instance, batch_idx, new_vocab):
        """Extract detailed information from a single query result"""
        # Convert log probability to regular probability
        probability = math.exp(-log_prob) if isinstance(log_prob, (int, float)) else math.exp(log_prob.item())
        
        # Calculate loss (negative log likelihood)
        loss = -log_prob if isinstance(log_prob, (int, float)) else -log_prob.item()
        
        # Extract object and bbox information
        predicted_object, predicted_bbox, bbox_id = self._extract_object_and_bbox_detailed(query_result, instance)
        
        return {
            'image_id': instance.metadata.get('image_id', 'unknown'),
            'query': str(instance.query.query),
            'full_result_query': str(query_result),  # Full bound query
            'predicted_object': predicted_object,
            'predicted_bbox': predicted_bbox,
            'bbox_id': bbox_id,  # NEW: Which bbox_id from scene graph
            'log_probability': log_prob if isinstance(log_prob, (int, float)) else log_prob.item(),
            'probability': probability,
            'loss': loss,
            'batch_idx': batch_idx,
            'new_vocab_count': len(new_vocab),
            'new_vocab_items': list(new_vocab),
            # Add ground truth info for easy comparison
            'ground_truth_object': instance.target[0] if hasattr(instance, 'target') and instance.target else None,
            'ground_truth_bbox_id': instance.target[1] if hasattr(instance, 'target') and instance.target else None,
            'is_correct_object': predicted_object == (instance.target[0] if hasattr(instance, 'target') and instance.target else None),
        }

    def _extract_object_and_bbox_detailed(self, query_result, instance):
        """Extract object name, bbox coordinates, and bbox_id from query result"""
        query_str = str(query_result)
        print(f"Parsing query result: {query_str}")
        
        # Method 1: Look for target(X) where X has been bound to an object
        target_match = re.search(r'target\(~?\(?([^)]+)\)?\)', query_str)
        
        if target_match:
            bound_object = target_match.group(1)
            # Remove soft term notation if present
            if bound_object.startswith('~'):
                bound_object = bound_object[1:]
            if bound_object.startswith('(') and bound_object.endswith(')'):
                bound_object = bound_object[1:-1]
            
            print(f"Found bound object from target(): {bound_object}")
            
            # Find corresponding bbox_id and coordinates in scene graph
            bbox_id, bbox_coords = self._find_bbox_details_for_object(bound_object, instance)
            return bound_object, bbox_coords, bbox_id
        
        print(f"Could not extract object from query result: {query_str}")
        return "unknown", None, None

    def _find_bbox_details_for_object(self, object_name, instance):
        """Find both bbox_id and coordinates for a given object name"""
        # Search through bounding boxes to find matching object
        for bbox_id, (obj_name, bbox_coords) in instance.scene_graph.bounding_boxes.items():
            if obj_name == object_name:
                return bbox_id, bbox_coords
        
        # If not found, might be an attribute - check attributes
        if hasattr(instance.scene_graph, 'attributes') and instance.scene_graph.attributes:
            for attr_id, attr_name in instance.scene_graph.attributes.items():
                if attr_name == object_name:
                    return attr_id, "attribute"
        
        print(f"No bbox found for object: {object_name}")
        return None, None

    def save_results_to_csv(self, results: List[Dict[str, Any]], output_path: str):
        """Save detailed results to CSV file"""
        print(f"Saving results to: {output_path}")
        
        # Create one row per result from each query
        flattened_results = []
        
        for summary in results:
            query_info = {
                'image_id': summary.get('image_id', 'unknown'),
                'original_query': summary.get('query', 'unknown'),
                'target_probability': summary.get('target_probability', 1.0),
                'total_results_count': summary.get('total_results', 0),
            }
            
            if summary.get('all_results') and len(summary['all_results']) > 0:
                # Add each individual result as a row
                for i, result in enumerate(summary['all_results']):
                    row = result.copy()
                    row.update(query_info)
                    row['result_rank'] = i + 1  # 1 = best result, 2 = second best, etc.
                    row['is_best_result'] = (i == 0)
                    flattened_results.append(row)
            else:
                # No results found - create error/empty row
                error_row = query_info.copy()
                error_row.update({
                    'predicted_object': 'NO_RESULTS',
                    'predicted_bbox': None,
                    'bbox_id': None,
                    'probability': 0.0,
                    'log_probability': float('-inf'),
                    'loss': float('inf'),
                    'result_rank': 0,
                    'is_best_result': False,
                    'error': 'No results found'
                })
                flattened_results.append(error_row)
        
        # Define comprehensive CSV columns
        columns = [
            # Query identification
            'image_id', 'original_query', 'batch_idx', 'target_probability',
            
            # Result details  
            'result_rank', 'is_best_result', 'full_result_query',
            'predicted_object', 'bbox_id', 'predicted_bbox',
            
            # Probabilities and metrics
            'probability', 'log_probability', 'loss',
            
            # Ground truth comparison
            'ground_truth_object', 'ground_truth_bbox_id', 'is_correct_object',
            
            # Summary info
            'total_results_count', 'new_vocab_count', 'new_vocab_items',
            
            # Error handling
            'error'
        ]
        
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            
            for row in flattened_results:
                # Handle list fields
                if isinstance(row.get('new_vocab_items'), list):
                    row['new_vocab_items'] = ';'.join(row['new_vocab_items'])
                if isinstance(row.get('predicted_bbox'), list):
                    row['predicted_bbox'] = str(row['predicted_bbox'])
                
                writer.writerow(row)
        
        print(f"Saved {len(flattened_results)} result rows to CSV")
        self._print_evaluation_summary(results, flattened_results)
    
    
    
    def _update_vocab_coverage(self, new_vocab):
        """Update vocabulary coverage statistics"""
        if new_vocab:
            self.vocab_coverage['new_constants'].update(new_vocab)
            self.vocab_coverage['total_new'] += len(new_vocab)
            self.vocab_coverage['instances_with_new_vocab'] += 1
            
            
    def _print_evaluation_summary(self, summaries, detailed_results):
        """Print evaluation summary statistics"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        print(f"Total instances: {len(summaries)}")
        print(f"Total individual results: {len(detailed_results)}")
        
        # Vocabulary coverage
        print(f"\nVocabulary Coverage:")
        print(f"  New constants encountered: {len(self.vocab_coverage['new_constants'])}")
        print(f"  Total new vocab instances: {self.vocab_coverage['total_new']}")
        print(f"  Instances with new vocab: {self.vocab_coverage['instances_with_new_vocab']}")
        
        # Results statistics
        results_with_answers = [r for r in detailed_results if r['predicted_object'] != 'NONE']
        print(f"\nResults Statistics:")
        print(f"  Instances with results: {len(results_with_answers)}")
        print(f"  Instances with no results: {len(detailed_results) - len(results_with_answers)}")
        
        if results_with_answers:
            probs = [r['probability'] for r in results_with_answers]
            print(f"  Average probability: {np.mean(probs):.4f}")
            print(f"  Max probability: {np.max(probs):.4f}")
            print(f"  Min probability: {np.min(probs):.4f}")
        
        # Accuracy if ground truth available
        if any('correct_prediction' in r for r in detailed_results):
            correct = sum(1 for r in detailed_results if r.get('correct_prediction', False))
            total = len([r for r in detailed_results if 'correct_prediction' in r])
            accuracy = correct / total if total > 0 else 0
            print(f"\nAccuracy: {accuracy:.3f} ({correct}/{total})")
        
        print("="*50)