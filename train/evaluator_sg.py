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
import os


def detect_bbox_format(bbox):
    """Detect if bbox is in [x,y,w,h] or [x1,y1,x2,y2] format"""
    if bbox is None or len(bbox) != 4:
        return "unknown"
    
    try:
        x1, y1, x2, y2 = [float(x) for x in bbox]
        # In [x1,y1,x2,y2] format, x2 > x1 and y2 > y1
        if x2 > x1 and y2 > y1:
            return "xyxy"
        # In [x,y,w,h] format, width and height should be positive
        elif x2 > 0 and y2 > 0:  # x2=width, y2=height
            return "xywh"
        else:
            return "unknown"
    except:
        return "unknown"


def convert_bbox_to_xyxy(bbox, format_hint=None):
    """Convert bbox to [x1,y1,x2,y2] format with auto-detection"""
    if bbox is None:
        return None
    
    try:
        bbox = [float(x) for x in bbox]
        
        # Auto-detect format if not provided
        if format_hint is None:
            format_hint = detect_bbox_format(bbox)
        
        if format_hint == "xywh":
            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = bbox
            return [x, y, x + w, y + h]
        elif format_hint == "xyxy":
            # Already in correct format
            return bbox
        else:
            print(f"Warning: Unknown bbox format for {bbox}, assuming xyxy")
            return bbox
            
    except (ValueError, IndexError, TypeError):
        print(f"Warning: Invalid bbox format - {bbox}")
        return None


def compute_iou(bbox1, bbox2):
    """
    Compute IoU between two bounding boxes with automatic format detection
    Handles both [x1,y1,x2,y2] and [x,y,w,h] formats
    """
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    # Convert both bboxes to xyxy format
    bbox1_xyxy = convert_bbox_to_xyxy(bbox1)
    bbox2_xyxy = convert_bbox_to_xyxy(bbox2)
    
    if bbox1_xyxy is None or bbox2_xyxy is None:
        return 0.0
    
    # Debug: Show conversion
    if detect_bbox_format(bbox1) == "xywh" or detect_bbox_format(bbox2) == "xywh":
        print(f"DEBUG: bbox1 {bbox1} ({detect_bbox_format(bbox1)}) -> {bbox1_xyxy}")
        print(f"DEBUG: bbox2 {bbox2} ({detect_bbox_format(bbox2)}) -> {bbox2_xyxy}")
    
    try:
        # Calculate intersection coordinates
        x1 = max(bbox1_xyxy[0], bbox2_xyxy[0])
        y1 = max(bbox1_xyxy[1], bbox2_xyxy[1])
        x2 = min(bbox1_xyxy[2], bbox2_xyxy[2])
        y2 = min(bbox1_xyxy[3], bbox2_xyxy[3])
        
        # Check if there's any intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate intersection area
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate individual areas
        area1 = (bbox1_xyxy[2] - bbox1_xyxy[0]) * (bbox1_xyxy[3] - bbox1_xyxy[1])
        area2 = (bbox2_xyxy[2] - bbox2_xyxy[0]) * (bbox2_xyxy[3] - bbox2_xyxy[1])
        
        # Calculate union
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
        
    except (ValueError, IndexError, TypeError):
        print(f"Warning: Invalid bbox format - bbox1: {bbox1_xyxy}, bbox2: {bbox2_xyxy}")
        return 0.0


class ReferringEvaluator(ReferringTrainer):
    def __init__(self, program, config, iou_threshold=0.3, **search_args):
        # Initialize without optimizer since we're not training
        self.program = program
        self.config = config
        self.search_args = search_args
        self.iou_threshold = iou_threshold  # NEW: IoU threshold for correct predictions
        self.vocab_coverage = {
            'new_constants': set(),
            'total_new': 0,
            'instances_with_new_vocab': 0
        }
        self.metadata_csv_path = "proof_metadata.csv"
        self._initialize_metadata_csv(self.metadata_csv_path)
        self.instance_counter = 0
        self.result_counter = 0
        print(f"🔍 Using IoU threshold: {self.iou_threshold}")
        
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
                    
                    # Create a properly structured error result
                    error_result = {
                        'image_id': getattr(instance, 'metadata', {}).get('image_id', 'unknown'),
                        'query': str(getattr(instance, 'query', 'Unknown')),
                        'batch_idx': batch_idx,
                        'target_probability': 1.0,
                        'new_vocab_count': 0,
                        'new_vocab_items': [],
                        'total_results': 0,
                        'best_predicted_object': 'ERROR',
                        'best_predicted_bbox': None,
                        'best_probability': 0.0,
                        'best_loss': float('inf'),
                        'best_iou': 0.0,  # NEW
                        'all_results': [{
                            'image_id': getattr(instance, 'metadata', {}).get('image_id', 'unknown'),
                            'query': str(getattr(instance, 'query', 'Unknown')),
                            'result_query': 'ERROR',
                            'predicted_object': 'ERROR',
                            'predicted_bbox': None,
                            'bbox_id': None,
                            'log_probability': float('-inf'),
                            'probability': 0.0,
                            'loss': float('inf'),
                            'iou': 0.0,  # NEW
                            'batch_idx': batch_idx,
                            'new_vocab_count': 0,
                            'new_vocab_items': [],
                        }]
                    }
                    all_results.append(error_result)
        
        return all_results

    
    def evaluate_single_instance(self, instance: DatasetInstance, batch_idx: int) -> Dict[str, Any]:
        """Evaluate a single instance and extract detailed results"""
        print(f"Evaluating instance with query: {instance.query}")
        
        self.instance_counter += 1
        self.result_counter = 0  # Reset for each instance

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
            results_dict, proof_steps, nb_proofs = self.program.query(instance.query.query, return_stats=True, **self.search_args)
        
        print(f"Raw results: {results_dict}")
        print(f"Number of results found: {len(results_dict)}")
        
        # FILTER: Remove any results that don't end with a number (bbox_id)
        filtered_results = {}
        for query_result, log_prob in results_dict.items():
            query_str = str(query_result)
            # Check if query ends with __bbox followed by a number
            if re.search(r'__bbox\d+$', query_str) or re.search(r'__\d+$', query_str):
                filtered_results[query_result] = log_prob
            else:
                print(f"FILTERED OUT: {query_str} (doesn't end with bbox number)")
        
        results_dict = filtered_results
        
        # Extract ground truth bbox coordinates from metadata
        target_bbox = instance.metadata.get('target_bbox', None)
        print(f"Target bbox from metadata: {target_bbox}")
        
        # Extract detailed results for all proven queries
        detailed_results = []
        result_counter = 0
        for query_result, log_prob in results_dict.items():
            result_counter += 1
            detailed_result = self._extract_enhanced_result_details(
                query_result, log_prob, instance, batch_idx, new_vocab, target_bbox
            )
            detailed_results.append(detailed_result)
            if hasattr(self.program, 'get_proof_metadata'):
                proof_metadata = self.program.get_proof_metadata(query_result)
                image_id = instance.metadata.get('image_id', 'unknown')
                self._append_metadata_to_csv(
                    self.metadata_csv_path, 
                    self.instance_counter, 
                    result_counter, 
                    image_id, 
                    query_result, 
                    proof_metadata
                )
        
        # If no results, create a default entry
        if not detailed_results:
            detailed_results.append(self._create_no_result_entry(instance, batch_idx, new_vocab, target_bbox))
        
        # Sort results by probability (highest first)
        detailed_results.sort(key=lambda x: x.get('probability', 0), reverse=True)
        
        print(f"After sorting, detailed_results has {len(detailed_results)} results:")
        for i, result in enumerate(detailed_results):
            print(f"  Rank {i+1}: {result['predicted_object']} (prob: {result['probability']:.6f}, IoU: {result['iou']:.3f})")
        
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
            'best_iou': best_result['iou'],  # NEW
            'all_results': detailed_results
        }
        
        # Add ground truth if available - UPDATED for generated scene graphs
        if hasattr(instance, 'target') and instance.target:
            summary_result['ground_truth_object'] = instance.target[0]
            # target[1] is now None for generated scene graphs
            summary_result['ground_truth_bbox_id'] = None  # Always None now
            summary_result['ground_truth_bbox'] = target_bbox  # From metadata
            
            # NEW: Use IoU-based correctness instead of bbox_id matching
            summary_result['correct_prediction'] = (best_result['iou'] >= self.iou_threshold)
        
        return summary_result
    

    def _create_no_result_entry(self, instance, batch_idx, new_vocab, target_bbox):
        """Create entry when no results are found"""
        return {
            'image_id': instance.metadata.get('image_id', 'unknown'),
            'query': str(instance.query.query),
            'result_query': 'NO_RESULT',
            'predicted_object': 'NONE',
            'predicted_bbox': None,
            'bbox_id': None,
            'log_probability': float('-inf'),
            'probability': 0.0,
            'loss': float('inf'),
            'iou': 0.0,  # NEW
            'batch_idx': batch_idx,
            'new_vocab_count': len(new_vocab),
            'new_vocab_items': list(new_vocab),
        }
    
    def _extract_enhanced_result_details(self, query_result, log_prob, instance, batch_idx, new_vocab, target_bbox):
        """Extract detailed information using enhanced proof metadata"""
        # Convert log probability to regular probability
        probability = math.exp(log_prob) if isinstance(log_prob, (int, float)) else math.exp(log_prob.item())
        loss = -log_prob if isinstance(log_prob, (int, float)) else -log_prob.item()
        
        # Get bbox_id from proof metadata
        bbox_id = None
        if hasattr(self.program, 'get_bbox_id_for_query_simple'):
            print(f"DEBUG: Calling method now...")
            bbox_id = self.program.get_bbox_id_for_query_simple(query_result)
            print(f"DEBUG: Method returned: {bbox_id}")
        
        # Get soft unifications
        soft_unifications = []
        if hasattr(self.program, 'get_soft_unifications_for_query'):
            soft_unifications = self.program.get_soft_unifications_for_query(query_result)
            print(f"DEBUG: Found {len(soft_unifications)} soft unifications")
        
        # Get variable bindings
        variable_bindings = {}
        if hasattr(self.program, 'get_variable_bindings_for_query'):
            variable_bindings = self.program.get_variable_bindings_for_query(query_result)
            print(f"DEBUG: Found variable bindings: {variable_bindings}")

        # Use bbox_id to look up exact object and bounding box
        predicted_object = "unknown"
        predicted_bbox = None
        
        if bbox_id:
            predicted_object, predicted_bbox = self._lookup_object_by_bbox_id(bbox_id, instance)
            print(f"Found object {predicted_object} with bbox {predicted_bbox}")
        else:
            # Fallback to existing extraction method
            predicted_object, bbox_id = self._extract_from_query_string(query_result, instance)
            if bbox_id:
                _, predicted_bbox = self._lookup_object_by_bbox_id(bbox_id, instance)
        
        # NEW: Compute IoU between predicted and target bboxes
        iou = compute_iou(predicted_bbox, target_bbox)
        print(f"IoU between predicted {predicted_bbox} and target {target_bbox}: {iou:.3f}")
        
        return {
            'image_id': instance.metadata.get('image_id', 'unknown'),
            'query': str(instance.query.query),
            'result_query': str(query_result),
            'predicted_object': predicted_object,
            'predicted_bbox': predicted_bbox,
            'bbox_id': bbox_id,
            'log_probability': log_prob if isinstance(log_prob, (int, float)) else log_prob.item(),
            'probability': probability,
            'loss': loss,
            'iou': iou,  # NEW: IoU score
            'batch_idx': batch_idx,
            'new_vocab_count': len(new_vocab),
            'new_vocab_items': list(new_vocab),
            'soft_unifications': soft_unifications,
            'variable_bindings': variable_bindings,
        }

    def _lookup_object_by_bbox_id(self, bbox_id, instance):
        """Look up object name and bbox coordinates from bbox_id"""
        if bbox_id in instance.scene_graph.bounding_boxes:
            obj_name, bbox_coords = instance.scene_graph.bounding_boxes[bbox_id]
            return obj_name, bbox_coords
        else:
            print(f"Warning: bbox_id {bbox_id} not found in scene graph")
            return "unknown", None
    
    def _update_vocab_coverage(self, new_vocab):
        """Update vocabulary coverage statistics"""
        if new_vocab:
            self.vocab_coverage['new_constants'].update(new_vocab)
            self.vocab_coverage['total_new'] += len(new_vocab)
            self.vocab_coverage['instances_with_new_vocab'] += 1

    def _initialize_metadata_csv(self, filepath):
        """Initialize metadata CSV with headers if it doesn't exist"""
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['instance_number', 'result_number', 'image_id', 'query_result', 'metadata', 'clauses'])

    def _append_metadata_to_csv(self, filepath, instance_number, result_number, image_id, query_result, metadata):
        """Append metadata row to CSV"""
        clauses = str(self.program.clauses) if hasattr(self.program, 'clauses') else 'N/A'
        
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([instance_number, result_number, image_id, str(query_result), str(metadata), clauses])

    def save_results_to_csv(self, results: List[Dict[str, Any]], output_path: str):
        """Save enhanced results to CSV file with IoU-based metrics"""
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
                        'is_best_result': (detail.get('bbox_id') == summary.get('best_bbox_id')),
                        'iou_threshold': self.iou_threshold,  # NEW
                    })
                    
                    # Add ground truth info if available
                    if 'ground_truth_object' in summary:
                        row['ground_truth_object'] = summary['ground_truth_object']
                        row['ground_truth_bbox_id'] = summary['ground_truth_bbox_id']  # Always None now
                        row['ground_truth_bbox'] = summary['ground_truth_bbox']
                        
                        # NEW: IoU-based correctness
                        row['correct_prediction'] = (detail.get('iou', 0) >= self.iou_threshold)
                        
                        # Additional accuracy metrics
                        row['object_name_match'] = (detail.get('predicted_object') == summary['ground_truth_object'])
                    
                    flattened_results.append(row)
            else:
                # No results case
                flattened_results.append(summary)
        
        print(f"Total flattened results to save: {len(flattened_results)}")
        
        # Enhanced CSV columns with IoU metrics
        columns = [
            # Basic info
            'image_id', 'batch_idx', 'query', 'result_query',
            # Predictions
            'predicted_object', 'predicted_bbox', 'bbox_id',  
            # Probabilities
            'target_probability', 'log_probability', 'probability', 'loss',
            # NEW: IoU metrics
            'iou', 'iou_threshold',
            # Soft unifications
            'num_soft_unifications', 'soft_unifications_summary', 'variable_bindings_summary',
            # Metadata
            'new_vocab_count', 'new_vocab_items',
            'total_results_for_query', 'is_best_result'
        ]
        
        # Add ground truth columns if available
        if any('ground_truth_object' in r for r in flattened_results):
            columns.extend([
                'ground_truth_object', 'ground_truth_bbox_id', 'ground_truth_bbox',  
                'correct_prediction',  # Now IoU-based
                'object_name_match'    # Object name-based accuracy (for comparison)
            ])
        
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for row in flattened_results:
                # Handle list/dict fields for CSV
                if 'soft_unifications' in row and isinstance(row['soft_unifications'], list):
                    unif_summary = []
                    for unif in row['soft_unifications']:
                        if isinstance(unif, dict):
                            if unif.get('type') == 'soft_fact' and unif.get('term1') and unif.get('term2'):
                                summary = f"{unif['term1']}~{unif['term2']}"
                                if unif.get('log_probability') is not None:
                                    summary += f"({unif['log_probability']:.3f})"
                            elif unif.get('type') == 'y_variable_binding':
                                summary = f"Y={unif.get('bbox_id', 'unknown')}"
                            else:
                                summary = f"{unif.get('type', 'unknown')}({unif.get('log_probability', 'N/A')})"
                            unif_summary.append(summary)
                    
                    row['soft_unifications_summary'] = ';'.join(unif_summary)
                    row['num_soft_unifications'] = len(row['soft_unifications'])
                    del row['soft_unifications']
                else:
                    row['soft_unifications_summary'] = ''
                    row['num_soft_unifications'] = 0
                
                # Format variable bindings for CSV
                if 'variable_bindings' in row and isinstance(row['variable_bindings'], dict):
                    binding_summary = []
                    for var, val in row['variable_bindings'].items():
                        binding_summary.append(f"{var}={val}")
                    row['variable_bindings_summary'] = ';'.join(binding_summary)
                    del row['variable_bindings']
                else:
                    row['variable_bindings_summary'] = ''
                
                # Handle other list fields
                if 'new_vocab_items' in row and isinstance(row['new_vocab_items'], list):
                    row['new_vocab_items'] = ';'.join(row['new_vocab_items'])
                
                # Handle bbox coordinates
                if 'predicted_bbox' in row and isinstance(row['predicted_bbox'], list):
                    row['predicted_bbox'] = str(row['predicted_bbox'])
                
                if 'ground_truth_bbox' in row and isinstance(row['ground_truth_bbox'], list):
                    row['ground_truth_bbox'] = str(row['ground_truth_bbox'])
                
                writer.writerow(row)
        
        # Enhanced summary statistics
        self._print_enhanced_evaluation_summary(results, flattened_results)

    def _print_enhanced_evaluation_summary(self, summaries, detailed_results):
        """Print enhanced evaluation summary with IoU-based metrics"""
        print("\n" + "="*60)
        print("ENHANCED EVALUATION SUMMARY (IoU-based)")
        print("="*60)
        
        print(f"Total instances: {len(summaries)}")
        print(f"Total individual results: {len(detailed_results)}")
        print(f"IoU threshold: {self.iou_threshold}")
        
        # Results statistics
        results_with_answers = [r for r in detailed_results if r.get('predicted_object') != 'NONE']
        results_with_bbox_id = [r for r in detailed_results if r.get('bbox_id') is not None]
        
        print(f"\nResults Statistics:")
        print(f"  Instances with results: {len(results_with_answers)}")
        print(f"  Instances with bbox_id: {len(results_with_bbox_id)}")
        print(f"  Instances with no results: {len(detailed_results) - len(results_with_answers)}")
        
        # NEW: IoU-based accuracy statistics
        if any('correct_prediction' in r for r in detailed_results):
            # IoU-based accuracy (main metric)
            iou_correct = sum(1 for r in detailed_results if r.get('correct_prediction', False))
            iou_total = len([r for r in detailed_results if 'correct_prediction' in r])
            iou_accuracy = iou_correct / iou_total if iou_total > 0 else 0
            
            # Object name-based accuracy (for comparison)
            name_correct = sum(1 for r in detailed_results if r.get('object_name_match', False))
            name_total = len([r for r in detailed_results if 'object_name_match' in r])
            name_accuracy = name_correct / name_total if name_total > 0 else 0
            
            print(f"\nAccuracy Metrics:")
            print(f"  IoU-based (IoU >= {self.iou_threshold}): {iou_accuracy:.3f} ({iou_correct}/{iou_total})")
            print(f"  Object name-based: {name_accuracy:.3f} ({name_correct}/{name_total})")
        
        # NEW: IoU distribution statistics
        iou_scores = [r.get('iou', 0) for r in detailed_results if r.get('iou') is not None]
        if iou_scores:
            print(f"\nIoU Distribution:")
            print(f"  Mean IoU: {np.mean(iou_scores):.3f}")
            print(f"  Median IoU: {np.median(iou_scores):.3f}")
            print(f"  Max IoU: {max(iou_scores):.3f}")
            print(f"  Min IoU: {min(iou_scores):.3f}")
            print(f"  IoU > 0: {sum(1 for iou in iou_scores if iou > 0)} / {len(iou_scores)}")
            print(f"  IoU >= 0.3: {sum(1 for iou in iou_scores if iou >= 0.3)} / {len(iou_scores)}")
            print(f"  IoU >= 0.5: {sum(1 for iou in iou_scores if iou >= 0.5)} / {len(iou_scores)}")
            print(f"  IoU >= 0.7: {sum(1 for iou in iou_scores if iou >= 0.7)} / {len(iou_scores)}")
        
        # Soft unification statistics
        soft_unif_counts = [r.get('num_soft_unifications', 0) for r in detailed_results if r.get('num_soft_unifications') is not None]
        if soft_unif_counts:
            print(f"\nSoft Unification Statistics:")
            print(f"  Total soft unifications: {sum(soft_unif_counts)}")
            print(f"  Average per result: {sum(soft_unif_counts) / len(soft_unif_counts):.2f}")
            print(f"  Max per result: {max(soft_unif_counts)}")
            print(f"  Results with soft unifications: {sum(1 for c in soft_unif_counts if c > 0)}")
        
        # bbox_id extraction success rate
        bbox_success_rate = len(results_with_bbox_id) / len(results_with_answers) if results_with_answers else 0
        print(f"\nExtraction Success Rate:")
        print(f"  bbox_id extraction: {bbox_success_rate:.3f} ({len(results_with_bbox_id)}/{len(results_with_answers)})")
        
        print("="*60)

    # Placeholder for missing methods that may be called
    def _extract_from_query_string(self, query_result, instance):
        """Fallback method for extracting object from query string"""
        query_str = str(query_result)
        # Simple regex to extract object from target(object) pattern
        target_match = re.search(r'target\(~?\(?([^)]+)\)?\)', query_str)
        
        if target_match:
            bound_object = target_match.group(1)
            # Remove soft term notation if present
            if bound_object.startswith('~'):
                bound_object = bound_object[1:]
            if bound_object.startswith('(') and bound_object.endswith(')'):
                bound_object = bound_object[1:-1]
            
            # Try to find bbox_id for this object
            for bbox_id, (obj_name, bbox_coords) in instance.scene_graph.bounding_boxes.items():
                if obj_name == bound_object:
                    return bound_object, bbox_id
        
        return "unknown", None