import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
import ast
import os
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Create output directory
os.makedirs('eval/plots', exist_ok=True)

def parse_bbox(bbox_str):
    """Parse bounding box string to list of coordinates"""
    if pd.isna(bbox_str) or bbox_str == '':
        return None
    try:
        # Remove brackets and split by comma
        coords = bbox_str.strip('[]').split(',')
        return [int(x.strip()) for x in coords]
    except:
        return None

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x, y, w, h]"""
    if box1 is None or box2 is None:
        return 0.0
    
    # Convert to [x1, y1, x2, y2] format
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    x1_2, y1_2, w2, h2 = box2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def handle_ties_and_calculate_accuracy(df, iou_threshold=0.5, probability_precision=3):
    """
    Calculate accuracy with proper tie handling - prioritize correct predictions in ties
    """
    print(f"\n🎯 CALCULATING ACCURACY WITH TIE HANDLING (IoU≥{iou_threshold}, prob precision={probability_precision})")
    
    # Update correct_prediction based on IoU threshold
    df['correct_prediction_iou'] = df['iou'] >= iou_threshold
    
    # Round probabilities to specified precision for tie detection
    df['prob_rounded'] = df['probability'].round(probability_precision)
    
    accuracy_results = {}
    tie_analysis = {}
    
    for batch_idx in df['batch_idx'].unique():
        batch_data = df[df['batch_idx'] == batch_idx].copy()
        
        # Skip if no valid predictions
        valid_predictions = batch_data[
            (batch_data['predicted_object'] != 'NONE') & 
            (batch_data['predicted_object'] != 'ERROR') &
            (batch_data['probability'] > 0)
        ]
        
        if len(valid_predictions) == 0:
            accuracy_results[batch_idx] = {'prediction': 'NO_RESULT', 'correct': False, 'tie_info': None}
            continue
        
        # Find maximum probability (rounded)
        max_prob = valid_predictions['prob_rounded'].max()
        
        # Get all predictions with maximum probability
        top_predictions = valid_predictions[valid_predictions['prob_rounded'] == max_prob].copy()
        
        # Analyze ties
        tie_info = {
            'num_tied': len(top_predictions),
            'max_prob': max_prob,
            'tied_objects': list(top_predictions['predicted_object'].unique()),
            'correct_in_tie': top_predictions['correct_prediction_iou'].any()
        }
        
        # If there's a tie, prioritize correct predictions
        if len(top_predictions) > 1:
            correct_in_tie = top_predictions[top_predictions['correct_prediction_iou'] == True]
            if len(correct_in_tie) > 0:
                # Use the first correct prediction in the tie
                chosen_prediction = correct_in_tie.iloc[0]
                tie_info['tie_resolution'] = 'correct_prioritized'
            else:
                # No correct prediction in tie, use first one (by original order)
                chosen_prediction = top_predictions.iloc[0]
                tie_info['tie_resolution'] = 'first_selected'
        else:
            chosen_prediction = top_predictions.iloc[0]
            tie_info['tie_resolution'] = 'no_tie'
        
        accuracy_results[batch_idx] = {
            'prediction': chosen_prediction['predicted_object'],
            'correct': chosen_prediction['correct_prediction_iou'],
            'tie_info': tie_info,
            'iou': chosen_prediction['iou'],
            'probability': chosen_prediction['probability']
        }
        
        tie_analysis[batch_idx] = tie_info
    
    # Calculate overall accuracy
    total_queries = len(accuracy_results)
    correct_predictions = sum(1 for r in accuracy_results.values() if r['correct'])
    accuracy = correct_predictions / total_queries * 100 if total_queries > 0 else 0
    
    # Analyze ties
    ties = [t for t in tie_analysis.values() if t['num_tied'] > 1]
    no_result = sum(1 for r in accuracy_results.values() if r['prediction'] == 'NO_RESULT')
    
    print(f"  Total queries: {total_queries}")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  No results: {no_result}")
    print(f"  Queries with ties: {len(ties)}")
    
    if len(ties) > 0:
        tie_sizes = [t['num_tied'] for t in ties]
        correct_in_ties = sum(1 for t in ties if t['correct_in_tie'])
        print(f"  Average tie size: {np.mean(tie_sizes):.2f}")
        print(f"  Max tie size: {max(tie_sizes)}")
        print(f"  Ties with correct answer: {correct_in_ties}/{len(ties)}")
        
        # Analyze tie resolution strategies
        resolution_strategies = [t['tie_resolution'] for t in ties]
        resolution_counts = Counter(resolution_strategies)
        print(f"  Tie resolution strategies: {dict(resolution_counts)}")
    
    return accuracy, accuracy_results, tie_analysis

def calculate_recall_at_k(df, k_values, iou_threshold=0.5, probability_precision=3):
    """Calculate Recall@K with proper tie handling"""
    print(f"\n📈 CALCULATING RECALL@K (IoU≥{iou_threshold}, prob precision={probability_precision})")
    
    # Update correct_prediction based on IoU threshold
    df['correct_prediction_iou'] = df['iou'] >= iou_threshold
    df['prob_rounded'] = df['probability'].round(probability_precision)
    
    recall_results = {}
    
    for k in k_values:
        correct_in_topk = 0
        total_with_results = 0
        
        for batch_idx in df['batch_idx'].unique():
            batch_data = df[df['batch_idx'] == batch_idx].copy()
            
            # Skip if no valid predictions
            valid_predictions = batch_data[
                (batch_data['predicted_object'] != 'NONE') & 
                (batch_data['predicted_object'] != 'ERROR') &
                (batch_data['probability'] > 0)
            ]
            
            if len(valid_predictions) == 0:
                continue
                
            total_with_results += 1
            
            # Sort by probability (descending), then by correctness (correct first for ties)
            valid_predictions = valid_predictions.sort_values(
                ['prob_rounded', 'correct_prediction_iou'], 
                ascending=[False, False]
            )
            
            # Get top-k predictions
            topk_predictions = valid_predictions.head(k)
            
            # Check if any are correct
            if topk_predictions['correct_prediction_iou'].any():
                correct_in_topk += 1
        
        recall_k = correct_in_topk / total_with_results * 100 if total_with_results > 0 else 0
        recall_results[k] = recall_k
        print(f"  Recall@{k}: {recall_k:.2f}% ({correct_in_topk}/{total_with_results})")
    
    return recall_results

def analyze_iou_threshold_effects(df, iou_thresholds):
    """Analyze how different IoU thresholds affect accuracy and recall"""
    print(f"\n🎛️ IoU THRESHOLD ANALYSIS")
    print("="*50)
    
    threshold_results = {}
    
    for threshold in iou_thresholds:
        print(f"\n--- IoU Threshold: {threshold} ---")
        
        # Calculate accuracy for this threshold
        accuracy, _, _ = handle_ties_and_calculate_accuracy(df, iou_threshold=threshold)
        
        # Calculate recall@k for this threshold (only up to 5)
        recall_results = calculate_recall_at_k(df, [1, 2, 3, 4, 5], iou_threshold=threshold)
        
        # Additional metrics
        df_temp = df.copy()
        df_temp['correct_at_threshold'] = df_temp['iou'] >= threshold
        
        # Count predictions above threshold
        above_threshold = df_temp[df_temp['correct_at_threshold'] == True]
        total_predictions = len(df_temp[df_temp['predicted_object'].notna() & 
                                      (df_temp['predicted_object'] != 'NONE') & 
                                      (df_temp['predicted_object'] != 'ERROR')])
        
        coverage = len(above_threshold) / total_predictions * 100 if total_predictions > 0 else 0
        
        threshold_results[threshold] = {
            'accuracy': accuracy,
            'recall': recall_results,
            'coverage': coverage,
            'predictions_above_threshold': len(above_threshold),
            'total_predictions': total_predictions
        }
        
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Coverage (% predictions above threshold): {coverage:.2f}%")
        print(f"  Predictions above threshold: {len(above_threshold)}/{total_predictions}")
    
    return threshold_results

def parse_soft_unifications(soft_unif_str):
    """Parse soft unifications string to list of unifications"""
    if pd.isna(soft_unif_str) or soft_unif_str == '':
        return []
    
    unifications = []
    parts = soft_unif_str.split(';')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Handle soft_fact entries
        if part.startswith('soft_fact'):
            match = re.match(r'soft_fact\(([^)]+)\)', part)
            if match:
                try:
                    neg_log_prob = float(match.group(1))
                    prob = np.exp(-neg_log_prob)  # Convert from negative log prob
                    unifications.append(('soft_fact', neg_log_prob, prob))
                except:
                    continue
        else:
            # Handle regular unifications like "rightOf~at(-5.952)"
            match = re.match(r'(.+?)\(([^)]+)\)$', part)
            if match:
                unif_type = match.group(1)
                score_str = match.group(2)
                try:
                    neg_log_prob = float(score_str)
                    prob = np.exp(-neg_log_prob)  # Convert from negative log prob
                    unifications.append((unif_type, neg_log_prob, prob))
                except:
                    continue
    
    return unifications

def analyze_bounding_box_characteristics(df, iou_threshold=0.3):
    """Analyze bounding box characteristics for successful predictions"""
    print(f"\n📦 BOUNDING BOX CHARACTERISTICS ANALYSIS (IoU≥{iou_threshold})")
    print("="*60)
    
    # Get successful predictions only
    successful_preds = df[(df['iou'] >= iou_threshold) & 
                         (df['parsed_pred_bbox'].notna()) & 
                         (df['parsed_gt_bbox'].notna())].copy()
    
    if len(successful_preds) == 0:
        print("No successful predictions found for bounding box analysis.")
        return None
    
    print(f"Analyzing {len(successful_preds)} successful predictions...")
    
    # Calculate bbox properties
    def calc_bbox_properties(bbox_list):
        if bbox_list is None or len(bbox_list) != 4:
            return None, None, None
        x, y, w, h = bbox_list
        return w, h, w * h
    
    # Calculate properties for predicted and ground truth bboxes
    pred_properties = []
    gt_properties = []
    
    for _, row in successful_preds.iterrows():
        pred_w, pred_h, pred_area = calc_bbox_properties(row['parsed_pred_bbox'])
        gt_w, gt_h, gt_area = calc_bbox_properties(row['parsed_gt_bbox'])
        
        if pred_w is not None and gt_w is not None:
            pred_properties.append((pred_w, pred_h, pred_area))
            gt_properties.append((gt_w, gt_h, gt_area))
    
    if not pred_properties:
        print("No valid bounding boxes found for analysis.")
        return None
    
    pred_widths = [p[0] for p in pred_properties]
    pred_heights = [p[1] for p in pred_properties]
    pred_areas = [p[2] for p in pred_properties]
    
    gt_widths = [p[0] for p in gt_properties]
    gt_heights = [p[1] for p in gt_properties]
    gt_areas = [p[2] for p in gt_properties]
    
    # Calculate statistics
    def calc_stats(values):
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }
    
    pred_width_stats = calc_stats(pred_widths)
    pred_height_stats = calc_stats(pred_heights)
    pred_area_stats = calc_stats(pred_areas)
    
    gt_width_stats = calc_stats(gt_widths)
    gt_height_stats = calc_stats(gt_heights)
    gt_area_stats = calc_stats(gt_areas)
    
    print(f"\nBounding Box Width Analysis:")
    print(f"  Predicted - Mean: {pred_width_stats['mean']:.1f}, Median: {pred_width_stats['median']:.1f}, Std: {pred_width_stats['std']:.1f}")
    print(f"  Ground Truth - Mean: {gt_width_stats['mean']:.1f}, Median: {gt_width_stats['median']:.1f}, Std: {gt_width_stats['std']:.1f}")
    print(f"  Difference (Pred - GT): {pred_width_stats['mean'] - gt_width_stats['mean']:.1f}")
    
    print(f"\nBounding Box Height Analysis:")
    print(f"  Predicted - Mean: {pred_height_stats['mean']:.1f}, Median: {pred_height_stats['median']:.1f}, Std: {pred_height_stats['std']:.1f}")
    print(f"  Ground Truth - Mean: {gt_height_stats['mean']:.1f}, Median: {gt_height_stats['median']:.1f}, Std: {gt_height_stats['std']:.1f}")
    print(f"  Difference (Pred - GT): {pred_height_stats['mean'] - gt_height_stats['mean']:.1f}")
    
    print(f"\nBounding Box Area Analysis:")
    print(f"  Predicted - Mean: {pred_area_stats['mean']:.0f}, Median: {pred_area_stats['median']:.0f}, Std: {pred_area_stats['std']:.0f}")
    print(f"  Ground Truth - Mean: {gt_area_stats['mean']:.0f}, Median: {gt_area_stats['median']:.0f}, Std: {gt_area_stats['std']:.0f}")
    print(f"  Difference (Pred - GT): {pred_area_stats['mean'] - gt_area_stats['mean']:.0f}")
    
    # Calculate ratios
    width_ratios = [p/g for p, g in zip(pred_widths, gt_widths)]
    height_ratios = [p/g for p, g in zip(pred_heights, gt_heights)]
    area_ratios = [p/g for p, g in zip(pred_areas, gt_areas)]
    
    print(f"\nSize Ratio Analysis (Predicted / Ground Truth):")
    print(f"  Width Ratio - Mean: {np.mean(width_ratios):.3f}, Median: {np.median(width_ratios):.3f}")
    print(f"  Height Ratio - Mean: {np.mean(height_ratios):.3f}, Median: {np.median(height_ratios):.3f}")
    print(f"  Area Ratio - Mean: {np.mean(area_ratios):.3f}, Median: {np.median(area_ratios):.3f}")
    
    # Analyze trends by object type
    print(f"\nTrends by Object Type:")
    object_bbox_stats = {}
    
    for obj_type in successful_preds['predicted_object'].unique():
        obj_data = successful_preds[successful_preds['predicted_object'] == obj_type]
        if len(obj_data) >= 3:  # Only analyze objects with 3+ instances
            
            obj_pred_widths = []
            obj_gt_widths = []
            obj_pred_areas = []
            obj_gt_areas = []
            
            for _, row in obj_data.iterrows():
                pred_w, pred_h, pred_area = calc_bbox_properties(row['parsed_pred_bbox'])
                gt_w, gt_h, gt_area = calc_bbox_properties(row['parsed_gt_bbox'])
                
                if pred_w is not None and gt_w is not None:
                    obj_pred_widths.append(pred_w)
                    obj_gt_widths.append(gt_w)
                    obj_pred_areas.append(pred_area)
                    obj_gt_areas.append(gt_area)
            
            if obj_pred_widths:
                width_ratio = np.mean(obj_pred_widths) / np.mean(obj_gt_widths)
                area_ratio = np.mean(obj_pred_areas) / np.mean(obj_gt_areas)
                
                object_bbox_stats[obj_type] = {
                    'count': len(obj_pred_widths),
                    'width_ratio': width_ratio,
                    'area_ratio': area_ratio,
                    'avg_pred_width': np.mean(obj_pred_widths),
                    'avg_gt_width': np.mean(obj_gt_widths)
                }
                
                print(f"  {obj_type} (n={len(obj_pred_widths)}): Width ratio={width_ratio:.3f}, Area ratio={area_ratio:.3f}")
    
    return {
        'pred_width_stats': pred_width_stats,
        'gt_width_stats': gt_width_stats,
        'pred_height_stats': pred_height_stats,
        'gt_height_stats': gt_height_stats,
        'pred_area_stats': pred_area_stats,
        'gt_area_stats': gt_area_stats,
        'width_ratios': width_ratios,
        'height_ratios': height_ratios,
        'area_ratios': area_ratios,
        'object_bbox_stats': object_bbox_stats,
        'successful_preds': successful_preds
    }

def analyze_successful_soft_unifications(df, iou_threshold=0.3):
    """Analyze soft unifications specifically for successful predictions"""
    print(f"\n🔗 SOFT UNIFICATIONS FOR SUCCESSFUL PREDICTIONS (IoU≥{iou_threshold})")
    print("="*60)
    
    # Get successful predictions only
    successful_preds = df[(df['iou'] >= iou_threshold) & 
                         (df['predicted_object'] != 'NONE') & 
                         (df['predicted_object'] != 'ERROR')].copy()
    
    if len(successful_preds) == 0:
        print("No successful predictions found for soft unification analysis.")
        return None
    
    print(f"Analyzing soft unifications from {len(successful_preds)} successful predictions...")
    
    # Parse soft unifications for successful predictions
    successful_preds['soft_unifications_parsed'] = successful_preds['soft_unifications_summary'].apply(parse_soft_unifications)
    
    # Collect all unifications from successful predictions
    successful_unifications = []
    for unifs in successful_preds['soft_unifications_parsed']:
        successful_unifications.extend(unifs)
    
    if not successful_unifications:
        print("No soft unifications found in successful predictions.")
        return None
    
    unif_types = [u[0] for u in successful_unifications]
    unif_probs = [u[2] for u in successful_unifications]
    
    unif_counter = Counter(unif_types)
    
    print(f"Total soft unifications in successful predictions: {len(successful_unifications)}")
    print(f"Unique unification types: {len(unif_counter)}")
    
    print(f"\nTop soft unifications in successful predictions:")
    for unif, count in unif_counter.most_common(15):
        # Calculate average probability for this unification type
        type_probs = [u[2] for u in successful_unifications if u[0] == unif]
        avg_prob = np.mean(type_probs) if type_probs else 0
        print(f"  {unif}: {count} times (avg prob: {avg_prob:.6f})")
    
    print(f"\nSoft unification probability statistics (successful predictions):")
    print(f"  Mean: {np.mean(unif_probs):.6f}")
    print(f"  Median: {np.median(unif_probs):.6f}")
    print(f"  Min: {np.min(unif_probs):.6f}")
    print(f"  Max: {np.max(unif_probs):.6f}")
    print(f"  Std: {np.std(unif_probs):.6f}")
    
    return {
        'successful_unif_counter': unif_counter,
        'successful_unif_probs': unif_probs,
        'successful_unifications': successful_unifications
    }

def analyze_comprehensive_results(df):
    """Perform comprehensive analysis of the neurosymbolic model results"""
    
    print("="*80)
    print("COMPREHENSIVE NEUROSYMBOLIC MODEL ANALYSIS")
    print("="*80)
    
    # Basic dataset statistics
    print(f"\n📊 DATASET OVERVIEW")
    total_queries = df['batch_idx'].nunique()
    print(f"Total predictions: {len(df)}")
    print(f"Total unique queries: {total_queries}")
    print(f"Average predictions per query: {len(df) / total_queries:.2f}")
    
    # Parse bounding boxes if not already done
    if 'parsed_pred_bbox' not in df.columns:
        df['parsed_pred_bbox'] = df['predicted_bbox'].apply(parse_bbox)
        df['parsed_gt_bbox'] = df['ground_truth_bbox'].apply(parse_bbox)
        
        # Recalculate IoU if needed
        df['bbox_iou'] = df.apply(lambda row: calculate_iou(row['parsed_pred_bbox'], row['parsed_gt_bbox']), axis=1)
    
    # =================== TIE ANALYSIS ===================
    print(f"\n🔗 PROBABILITY TIE ANALYSIS")
    print("="*40)
    
    # Analyze probability distributions and ties
    df['prob_rounded_3'] = df['probability'].round(3)
    
    tie_batches = []
    for batch_idx in df['batch_idx'].unique():
        batch_data = df[df['batch_idx'] == batch_idx]
        valid_predictions = batch_data[
            (batch_data['predicted_object'] != 'NONE') & 
            (batch_data['predicted_object'] != 'ERROR') &
            (batch_data['probability'] > 0)
        ]
        
        if len(valid_predictions) > 1:
            max_prob = valid_predictions['prob_rounded_3'].max()
            tied_predictions = valid_predictions[valid_predictions['prob_rounded_3'] == max_prob]
            
            if len(tied_predictions) > 1:
                tie_batches.append({
                    'batch_idx': batch_idx,
                    'num_tied': len(tied_predictions),
                    'max_prob': max_prob,
                    'objects': list(tied_predictions['predicted_object']),
                    'correct_in_tie': tied_predictions['correct_prediction'].any()
                })
    
    print(f"Queries with probability ties: {len(tie_batches)}/{total_queries} ({len(tie_batches)/total_queries*100:.1f}%)")
    if tie_batches:
        tie_sizes = [t['num_tied'] for t in tie_batches]
        print(f"Average tie size: {np.mean(tie_sizes):.2f}")
        print(f"Largest tie: {max(tie_sizes)} predictions")
        
        correct_in_ties = sum(1 for t in tie_batches if t['correct_in_tie'])
        print(f"Ties containing correct answer: {correct_in_ties}/{len(tie_batches)} ({correct_in_ties/len(tie_batches)*100:.1f}%)")
        
        # Show some examples
        print(f"\nExample ties:")
        for i, tie in enumerate(tie_batches[:3]):
            print(f"  Batch {tie['batch_idx']}: {tie['num_tied']} predictions at prob={tie['max_prob']:.3f}")
            print(f"    Objects: {tie['objects']}")
            print(f"    Contains correct: {tie['correct_in_tie']}")
    
    # =================== IoU THRESHOLD ANALYSIS ===================
    iou_thresholds = [0.3, 0.5, 0.7]  # Focus on these three thresholds
    threshold_results = analyze_iou_threshold_effects(df, iou_thresholds)
    
    # =================== DETAILED RECALL@K ANALYSIS ===================
    print(f"\n📈 COMPREHENSIVE RECALL@K ANALYSIS")
    print("="*50)
    
    k_values = [1, 2, 3, 4, 5]  # Only up to 5 as requested
    
    # Recall@K for different IoU thresholds
    for iou_thresh in iou_thresholds:
        print(f"\n--- Recall@K for IoU≥{iou_thresh} ---")
        recall_results = calculate_recall_at_k(df, k_values, iou_threshold=iou_thresh)
    
    # =================== BOUNDING BOX CHARACTERISTICS ANALYSIS ===================
    bbox_analysis = analyze_bounding_box_characteristics(df, iou_threshold=0.3)
    
    # =================== SUCCESSFUL PREDICTIONS SOFT UNIFICATIONS ===================
    successful_unif_analysis = analyze_successful_soft_unifications(df, iou_threshold=0.3)
    
    # =================== PROBABILITY DISTRIBUTION ANALYSIS ===================
    print(f"\n📊 DETAILED PROBABILITY ANALYSIS")
    print("="*40)
    
    valid_probs = df[df['probability'] > 0]['probability']
    
    print(f"Probability Statistics:")
    print(f"  Count: {len(valid_probs)}")
    print(f"  Mean: {valid_probs.mean():.6f}")
    print(f"  Median: {valid_probs.median():.6f}")
    print(f"  Std: {valid_probs.std():.6f}")
    print(f"  Min: {valid_probs.min():.6f}")
    print(f"  Max: {valid_probs.max():.6f}")
    print(f"  25th percentile: {valid_probs.quantile(0.25):.6f}")
    print(f"  75th percentile: {valid_probs.quantile(0.75):.6f}")
    print(f"  90th percentile: {valid_probs.quantile(0.90):.6f}")
    print(f"  99th percentile: {valid_probs.quantile(0.99):.6f}")
    
    # Probability ranges
    prob_ranges = [
        (0, 1e-6, "Extremely low (<1e-6)"),
        (1e-6, 1e-5, "Very low (1e-6 to 1e-5)"),
        (1e-5, 1e-4, "Low (1e-5 to 1e-4)"),
        (1e-4, 1e-3, "Low-medium (1e-4 to 1e-3)"),
        (1e-3, 1e-2, "Medium (1e-3 to 1e-2)"),
        (1e-2, 0.1, "High-medium (1e-2 to 0.1)"),
        (0.1, 1.0, "High (>0.1)")
    ]
    
    print(f"\nProbability Distribution:")
    for low, high, label in prob_ranges:
        count = sum((valid_probs >= low) & (valid_probs < high))
        print(f"  {label}: {count}/{len(valid_probs)} ({count/len(valid_probs)*100:.1f}%)")
    
    # =================== OBJECT TYPE ANALYSIS ===================
    print(f"\n🏷️ DETAILED OBJECT TYPE ANALYSIS")
    print("="*40)
    
    # Get best results for analysis
    best_results = []
    for batch_idx in df['batch_idx'].unique():
        batch_data = df[df['batch_idx'] == batch_idx]
        
        valid_predictions = batch_data[
            (batch_data['predicted_object'] != 'NONE') & 
            (batch_data['predicted_object'] != 'ERROR') &
            (batch_data['probability'] > 0)
        ]
        
        if len(valid_predictions) > 0:
            # Get highest probability prediction
            best_idx = valid_predictions['probability'].idxmax()
            best_results.append(valid_predictions.loc[best_idx])
        else:
            # No valid predictions
            no_result_row = batch_data.iloc[0].copy()
            no_result_row['predicted_object'] = 'NO_RESULT'
            best_results.append(no_result_row)
    
    best_df = pd.DataFrame(best_results)
    
    # Only proceed with analysis if we have results
    if len(best_df) == 0:
        print("No best results found - skipping object analysis")
        gt_objects = pd.Series(dtype=object)
        pred_objects = pd.Series(dtype=object)
    else:
        # Ground truth object distribution
        gt_objects = best_df['ground_truth_object'].value_counts()
        # print(f"\nGround Truth Object Distribution:")
        # for obj, count in gt_objects.items():
        #     print(f"  {obj}: {count}")
        
        # Predicted object distribution
        pred_objects = best_df[best_df['predicted_object'].notna()]['predicted_object'].value_counts()
        print(f"\nPredicted Object Distribution:")
        for obj, count in pred_objects.items():
            print(f"  {obj}: {count}")
    
    # Object-level accuracy
    if len(gt_objects) > 0:
        print(f"\nPer-Object Accuracy Analysis:")
        for gt_obj in gt_objects.index:
            gt_subset = best_df[best_df['ground_truth_object'] == gt_obj]
            
            # Count different types of predictions for this GT object
            pred_dist = gt_subset['predicted_object'].value_counts()
            
            # Handle object_name_match column gracefully
            if 'object_name_match' in gt_subset.columns:
                correct_count = len(gt_subset[gt_subset['object_name_match'] == True])
            else:
                # Fallback: check if predicted object matches ground truth object
                correct_count = len(gt_subset[gt_subset['predicted_object'] == gt_obj])
            
            total_count = len(gt_subset)
            
            # print(f"\n  Ground Truth '{gt_obj}' ({total_count} instances):")
            # print(f"    Correct name predictions: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
            # # print(f"    Prediction distribution:")
            # for pred_obj, count in pred_dist.items():
            #     print(f"      {pred_obj}: {count} ({count/total_count*100:.1f}%)")
    else:
        print(f"\nNo ground truth objects found for analysis")
    
    # =================== QUERY COMPLEXITY ANALYSIS ===================
    print(f"\n🧮 QUERY COMPLEXITY ANALYSIS")
    print("="*40)
    
    # Count expressions in queries
    def count_expressions(query_str):
        if pd.isna(query_str):
            return 0
        return query_str.count('expression(')
    
    df['query_complexity'] = df['query'].apply(count_expressions)
    
    complexity_analysis = {}
    for complexity in sorted(df['query_complexity'].unique()):
        # Get best results for each complexity level
        complex_batches = df[df['query_complexity'] == complexity]['batch_idx'].unique()
        
        complex_best = []
        for batch_idx in complex_batches:
            batch_data = df[df['batch_idx'] == batch_idx]
            
            valid_predictions = batch_data[
                (batch_data['predicted_object'] != 'NONE') & 
                (batch_data['predicted_object'] != 'ERROR') &
                (batch_data['probability'] > 0)
            ]
            
            if len(valid_predictions) > 0:
                best_idx = valid_predictions['probability'].idxmax()
                complex_best.append(valid_predictions.loc[best_idx])
        
        if complex_best:
            complex_df = pd.DataFrame(complex_best)
            
            total_queries = len(complex_df)
            
            # Handle object_name_match column gracefully
            if 'object_name_match' in complex_df.columns:
                correct_obj = len(complex_df[complex_df['object_name_match'] == True])
            else:
                # Fallback: check if predicted object matches ground truth object
                correct_obj = len(complex_df[complex_df['predicted_object'] == complex_df['ground_truth_object']])
            
            correct_iou_05 = len(complex_df[complex_df['iou'] >= 0.5])
            correct_iou_03 = len(complex_df[complex_df['iou'] >= 0.3])
            
            avg_prob = complex_df['probability'].mean()
            avg_iou = complex_df['iou'].mean()
            
            complexity_analysis[complexity] = {
                'total_queries': total_queries,
                'correct_obj': correct_obj,
                'correct_iou_05': correct_iou_05,
                'correct_iou_03': correct_iou_03,
                'avg_prob': avg_prob,
                'avg_iou': avg_iou
            }
            
            print(f"\nComplexity {complexity} expressions ({total_queries} queries):")
            print(f"  Object name accuracy: {correct_obj/total_queries*100:.1f}% ({correct_obj}/{total_queries})")
            print(f"  IoU≥0.5 accuracy: {correct_iou_05/total_queries*100:.1f}% ({correct_iou_05}/{total_queries})")
            print(f"  IoU≥0.3 accuracy: {correct_iou_03/total_queries*100:.1f}% ({correct_iou_03}/{total_queries})")
            print(f"  Average probability: {avg_prob:.6f}")
            print(f"  Average IoU: {avg_iou:.3f}")
    
    # =================== SOFT UNIFICATION ANALYSIS ===================
    print(f"\n🔗 SOFT UNIFICATION DETAILED ANALYSIS")
    print("="*40)
    
    # Parse soft unifications
    df['soft_unifications_parsed'] = df['soft_unifications_summary'].apply(parse_soft_unifications)
    
    all_unifications = []
    for unifs in df['soft_unifications_parsed']:
        all_unifications.extend(unifs)
    
    unif_types = [u[0] for u in all_unifications]
    unif_probs = [u[2] for u in all_unifications]
    
    unif_counter = Counter(unif_types)
    
    print(f"Total soft unifications: {len(all_unifications)}")
    print(f"Unique unification types: {len(unif_counter)}")
    
    print(f"\nTop 20 most common soft unifications:")
    for unif, count in unif_counter.most_common(20):
        # Calculate average probability for this unification type
        type_probs = [u[2] for u in all_unifications if u[0] == unif]
        avg_prob = np.mean(type_probs) if type_probs else 0
        print(f"  {unif}: {count} times (avg prob: {avg_prob:.6f})")
    
    print(f"\nSoft unification probability statistics:")
    print(f"  Mean: {np.mean(unif_probs):.6f}")
    print(f"  Median: {np.median(unif_probs):.6f}")
    print(f"  Min: {np.min(unif_probs):.6f}")
    print(f"  Max: {np.max(unif_probs):.6f}")
    
    # =================== CREATE ENHANCED VISUALIZATIONS ===================
    print(f"\n📊 CREATING ENHANCED VISUALIZATIONS...")
    
    # Debug: Print key statistics before visualization
    print(f"Debug - Data counts:")
    print(f"  Total dataframe rows: {len(df)}")
    print(f"  Total unique batches: {total_queries}")
    print(f"  Best results length: {len(best_results)}")
    
    if len(best_results) > 0:
        best_results_df_debug = pd.DataFrame(best_results)
        print(f"  IoU≥0.7: {len(best_results_df_debug[best_results_df_debug['iou'] >= 0.7])}")
        print(f"  IoU≥0.5: {len(best_results_df_debug[best_results_df_debug['iou'] >= 0.5])}")
        print(f"  IoU≥0.3: {len(best_results_df_debug[best_results_df_debug['iou'] >= 0.3])}")
        print(f"  NO_RESULT: {len(best_results_df_debug[best_results_df_debug['predicted_object'] == 'NO_RESULT'])}")
    
    # 1. IoU Threshold Analysis Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy vs IoU threshold
    thresholds = list(threshold_results.keys())
    accuracies = [threshold_results[t]['accuracy'] for t in thresholds]
    
    ax1.plot(thresholds, accuracies, 'bo-', linewidth=3, markersize=10)
    ax1.set_xlabel('IoU Threshold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy vs IoU Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.25, 0.75)
    for i, (thresh, acc) in enumerate(zip(thresholds, accuracies)):
        ax1.annotate(f'{acc:.1f}%', (thresh, acc), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Coverage vs IoU threshold
    coverages = [threshold_results[t]['coverage'] for t in thresholds]
    ax2.plot(thresholds, coverages, 'ro-', linewidth=3, markersize=10)
    ax2.set_xlabel('IoU Threshold')
    ax2.set_ylabel('Coverage (%)')
    ax2.set_title('Coverage vs IoU Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.25, 0.75)
    for i, (thresh, cov) in enumerate(zip(thresholds, coverages)):
        ax2.annotate(f'{cov:.1f}%', (thresh, cov), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Recall@K comparison for different IoU thresholds
    k_vals = [1, 2, 3, 4, 5]
    
    for iou_thresh in thresholds:
        recalls = []
        for k in k_vals:
            recall_res = calculate_recall_at_k(df, [k], iou_threshold=iou_thresh)
            recalls.append(recall_res[k])
        ax3.plot(k_vals, recalls, 'o-', linewidth=2, label=f'IoU≥{iou_thresh}', markersize=8)
    
    ax3.set_xlabel('K')
    ax3.set_ylabel('Recall@K (%)')
    ax3.set_title('Recall@K for Different IoU Thresholds')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(k_vals)
    
    # Probability tie analysis
    tie_counts = Counter([len(batch['objects']) for batch in tie_batches])
    if tie_counts:
        sizes = list(tie_counts.keys())
        counts = list(tie_counts.values())
        ax4.bar(sizes, counts, alpha=0.7, color='purple')
        ax4.set_xlabel('Tie Size (Number of Tied Predictions)')
        ax4.set_ylabel('Number of Queries')
        ax4.set_title('Distribution of Probability Tie Sizes')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eval/plots/enhanced_iou_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bounding Box Analysis Visualizations
    if bbox_analysis:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Width comparison
        ax1.hist(bbox_analysis['successful_preds']['parsed_pred_bbox'].apply(lambda x: x[2] if x else 0), 
                bins=30, alpha=0.7, label='Predicted', color='blue')
        ax1.hist(bbox_analysis['successful_preds']['parsed_gt_bbox'].apply(lambda x: x[2] if x else 0), 
                bins=30, alpha=0.7, label='Ground Truth', color='red')
        ax1.set_xlabel('Width (pixels)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Bounding Box Width Distribution (Successful Predictions)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Height comparison
        ax2.hist(bbox_analysis['successful_preds']['parsed_pred_bbox'].apply(lambda x: x[3] if x else 0), 
                bins=30, alpha=0.7, label='Predicted', color='blue')
        ax2.hist(bbox_analysis['successful_preds']['parsed_gt_bbox'].apply(lambda x: x[3] if x else 0), 
                bins=30, alpha=0.7, label='Ground Truth', color='red')
        ax2.set_xlabel('Height (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Bounding Box Height Distribution (Successful Predictions)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Width ratio distribution
        ax3.hist(bbox_analysis['width_ratios'], bins=30, alpha=0.7, color='green')
        ax3.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Ratio')
        ax3.set_xlabel('Width Ratio (Predicted / Ground Truth)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Width Ratio Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Area ratio distribution
        ax4.hist(bbox_analysis['area_ratios'], bins=30, alpha=0.7, color='orange')
        ax4.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Ratio')
        ax4.set_xlabel('Area Ratio (Predicted / Ground Truth)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Area Ratio Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('eval/plots/bounding_box_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Object-specific bounding box analysis
        if bbox_analysis['object_bbox_stats']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            objects = list(bbox_analysis['object_bbox_stats'].keys())
            width_ratios = [bbox_analysis['object_bbox_stats'][obj]['width_ratio'] for obj in objects]
            area_ratios = [bbox_analysis['object_bbox_stats'][obj]['area_ratio'] for obj in objects]
            counts = [bbox_analysis['object_bbox_stats'][obj]['count'] for obj in objects]
            
            # Width ratios by object type
            bars1 = ax1.bar(range(len(objects)), width_ratios, alpha=0.7, color='skyblue')
            ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Ratio')
            ax1.set_xlabel('Object Type')
            ax1.set_ylabel('Width Ratio (Pred/GT)')
            ax1.set_title('Width Ratio by Object Type')
            ax1.set_xticks(range(len(objects)))
            ax1.set_xticklabels(objects, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add count annotations
            for i, (bar, count) in enumerate(zip(bars1, counts)):
                ax1.annotate(f'n={count}', (bar.get_x() + bar.get_width()/2, bar.get_height()), 
                           textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
            
            # Area ratios by object type
            bars2 = ax2.bar(range(len(objects)), area_ratios, alpha=0.7, color='lightcoral')
            ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Ratio')
            ax2.set_xlabel('Object Type')
            ax2.set_ylabel('Area Ratio (Pred/GT)')
            ax2.set_title('Area Ratio by Object Type')
            ax2.set_xticks(range(len(objects)))
            ax2.set_xticklabels(objects, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add count annotations
            for i, (bar, count) in enumerate(zip(bars2, counts)):
                ax2.annotate(f'n={count}', (bar.get_x() + bar.get_width()/2, bar.get_height()), 
                           textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig('eval/plots/object_specific_bbox_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Soft Unification Comparison (All vs Successful)
    if successful_unif_analysis:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Top unifications in successful predictions
        top_successful_unifs = successful_unif_analysis['successful_unif_counter'].most_common(15)
        if top_successful_unifs:
            unif_names = [u[0] for u in top_successful_unifs]
            unif_counts = [u[1] for u in top_successful_unifs]
            
            ax1.barh(range(len(unif_names)), unif_counts, alpha=0.7, color='green')
            ax1.set_yticks(range(len(unif_names)))
            ax1.set_yticklabels(unif_names, fontsize=8)
            ax1.set_xlabel('Frequency')
            ax1.set_title('Top 15 Soft Unifications (Successful Predictions Only)')
            ax1.grid(True, alpha=0.3)
        
        # Probability distribution comparison
        all_unif_probs = [u[2] for u in all_unifications]
        successful_unif_probs = successful_unif_analysis['successful_unif_probs']
        
        ax2.hist(np.log10(np.array(all_unif_probs) + 1e-10), bins=50, alpha=0.7, 
                label='All Predictions', color='blue')
        ax2.hist(np.log10(np.array(successful_unif_probs) + 1e-10), bins=50, alpha=0.7, 
                label='Successful Predictions', color='green')
        ax2.set_xlabel('Log10(Soft Unification Probability)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Soft Unification Probability Distribution Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Top unifications overall (for comparison)
        top_overall_unifs = unif_counter.most_common(15)
        if top_overall_unifs:
            unif_names_all = [u[0] for u in top_overall_unifs]
            unif_counts_all = [u[1] for u in top_overall_unifs]
            
            ax3.barh(range(len(unif_names_all)), unif_counts_all, alpha=0.7, color='blue')
            ax3.set_yticks(range(len(unif_names_all)))
            ax3.set_yticklabels(unif_names_all, fontsize=8)
            ax3.set_xlabel('Frequency')
            ax3.set_title('Top 15 Soft Unifications (All Predictions)')
            ax3.grid(True, alpha=0.3)
        
        # Success rate by soft unification type
        unif_success_rates = {}
        for unif_type in unif_counter.keys():
            if unif_counter[unif_type] >= 5:  # Only analyze unifications that appear 5+ times
                total_with_unif = 0
                successful_with_unif = 0
                
                for _, row in df.iterrows():
                    if unif_type in str(row['soft_unifications_summary']):
                        total_with_unif += 1
                        if row['iou'] >= 0.3:
                            successful_with_unif += 1
                
                if total_with_unif > 0:
                    success_rate = successful_with_unif / total_with_unif * 100
                    unif_success_rates[unif_type] = success_rate
        
        if unif_success_rates:
            # Sort by success rate
            sorted_unifs = sorted(unif_success_rates.items(), key=lambda x: x[1], reverse=True)[:15]
            
            unif_names_sr = [u[0] for u in sorted_unifs]
            success_rates = [u[1] for u in sorted_unifs]
            
            ax4.barh(range(len(unif_names_sr)), success_rates, alpha=0.7, color='purple')
            ax4.set_yticks(range(len(unif_names_sr)))
            ax4.set_yticklabels(unif_names_sr, fontsize=8)
            ax4.set_xlabel('Success Rate (%)')
            ax4.set_title('Success Rate by Soft Unification Type (≥5 occurrences)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('eval/plots/soft_unification_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Comprehensive Performance Dashboard
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
    
    # Overall accuracy pie chart - fix negative values issue
    best_results_df = pd.DataFrame(best_results)
    
    # Count each category properly
    iou_07_plus = len(best_results_df[best_results_df['iou'] >= 0.7])
    iou_05_to_07 = len(best_results_df[(best_results_df['iou'] >= 0.5) & (best_results_df['iou'] < 0.7)])
    iou_03_to_05 = len(best_results_df[(best_results_df['iou'] >= 0.3) & (best_results_df['iou'] < 0.5)])
    no_results = len(best_results_df[best_results_df['predicted_object'] == 'NO_RESULT'])
    iou_below_03 = total_queries - iou_07_plus - iou_05_to_07 - iou_03_to_05 - no_results
    
    # Ensure no negative values
    iou_below_03 = max(0, iou_below_03)
    
    labels = [f'IoU≥0.7 ({iou_07_plus})', f'0.5≤IoU<0.7 ({iou_05_to_07})', 
              f'0.3≤IoU<0.5 ({iou_03_to_05})', f'IoU<0.3 ({iou_below_03})', 
              f'No Result ({no_results})']
    sizes = [iou_07_plus, iou_05_to_07, iou_03_to_05, iou_below_03, no_results]
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#95a5a6']
    
    # Only create pie chart if we have positive sizes
    if all(s >= 0 for s in sizes) and sum(sizes) > 0:
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Performance Distribution by IoU Thresholds')
    else:
        ax1.text(0.5, 0.5, 'Insufficient data\nfor pie chart', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Performance Distribution by IoU Thresholds')
    
    # Query complexity heatmap
    if complexity_analysis:
        complexity_data = []
        complexity_labels = []
        for complexity in sorted(complexity_analysis.keys()):
            stats = complexity_analysis[complexity]
            complexity_data.append([
                stats['correct_obj']/stats['total_queries']*100,
                stats['correct_iou_03']/stats['total_queries']*100,
                stats['correct_iou_05']/stats['total_queries']*100
            ])
            complexity_labels.append(f'{complexity} expr')
        
        if complexity_data:
            complexity_array = np.array(complexity_data).T
            im = ax2.imshow(complexity_array, cmap='RdYlGn', aspect='auto')
            ax2.set_xticks(range(len(complexity_labels)))
            ax2.set_xticklabels(complexity_labels)
            ax2.set_yticks(range(3))
            ax2.set_yticklabels(['Object Name', 'IoU≥0.3', 'IoU≥0.5'])
            ax2.set_title('Accuracy Heatmap by Query Complexity')
            
            # Add text annotations
            for i in range(3):
                for j in range(len(complexity_labels)):
                    ax2.text(j, i, f'{complexity_array[i, j]:.1f}%', 
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Probability vs IoU scatter plot
    valid_data = df[(df['probability'] > 0) & (df['iou'] >= 0)]
    if len(valid_data) > 0:
        scatter = ax3.scatter(valid_data['probability'], valid_data['iou'], 
                            alpha=0.6, c=valid_data['iou'], cmap='RdYlGn')
        ax3.set_xlabel('Probability')
        ax3.set_ylabel('IoU')
        ax3.set_title('Probability vs IoU Correlation')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='IoU')
    
    # Object type performance
    if len(gt_objects) > 0:
        obj_performance = []
        obj_names_perf = []
        
        for gt_obj in gt_objects.index[:10]:
            gt_subset = best_results_df[best_results_df['ground_truth_object'] == gt_obj]
            if len(gt_subset) >= 3:
                # Handle potential missing columns gracefully
                if 'object_name_match' in gt_subset.columns:
                    name_acc = len(gt_subset[gt_subset['object_name_match'] == True]) / len(gt_subset) * 100
                else:
                    # Fallback: check if predicted object matches ground truth object
                    name_acc = len(gt_subset[gt_subset['predicted_object'] == gt_obj]) / len(gt_subset) * 100
                
                iou_acc = len(gt_subset[gt_subset['iou'] >= 0.3]) / len(gt_subset) * 100
                
                obj_performance.append([name_acc, iou_acc])
                obj_names_perf.append(f'{gt_obj}\n(n={len(gt_subset)})')
        
        if obj_performance:
            x = np.arange(len(obj_names_perf))
            width = 0.35
            
            name_accs = [p[0] for p in obj_performance]
            iou_accs = [p[1] for p in obj_performance]
            
            ax4.bar(x - width/2, name_accs, width, label='Object Name', alpha=0.7, color='skyblue')
            ax4.bar(x + width/2, iou_accs, width, label='IoU≥0.3', alpha=0.7, color='lightcoral')
            
            ax4.set_xlabel('Object Type')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_title('Performance by Object Type')
            ax4.set_xticks(x)
            ax4.set_xticklabels(obj_names_perf, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor object analysis', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Performance by Object Type')
    else:
        ax4.text(0.5, 0.5, 'No object data\navailable', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance by Object Type')
    
    # Recall curves comparison
    recall_data = {}
    for thresh in [0.3, 0.5, 0.7]:
        recalls = []
        for k in [1, 2, 3, 4, 5]:
            recall_res = calculate_recall_at_k(df, [k], iou_threshold=thresh)
            recalls.append(recall_res[k])
        recall_data[thresh] = recalls
    
    for thresh, recalls in recall_data.items():
        ax5.plot([1, 2, 3, 4, 5], recalls, 'o-', linewidth=2, 
                label=f'IoU≥{thresh}', markersize=8)
    
    ax5.set_xlabel('K')
    ax5.set_ylabel('Recall@K (%)')
    ax5.set_title('Recall@K Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks([1, 2, 3, 4, 5])
    
    # Prediction confidence distribution
    confidence_ranges = [
        (0, 1e-5, 'Very Low\n(<1e-5)'),
        (1e-5, 1e-4, 'Low\n(1e-5 to 1e-4)'),
        (1e-4, 1e-3, 'Medium\n(1e-4 to 1e-3)'),
        (1e-3, 1e-2, 'High\n(1e-3 to 1e-2)'),
        (1e-2, 1, 'Very High\n(>1e-2)')
    ]
    
    range_counts = []
    range_labels = []
    
    for low, high, label in confidence_ranges:
        count = len(valid_probs[(valid_probs >= low) & (valid_probs < high)])
        range_counts.append(count)
        range_labels.append(label)
    
    ax6.bar(range(len(range_labels)), range_counts, alpha=0.7, color='mediumpurple')
    ax6.set_xlabel('Confidence Range')
    ax6.set_ylabel('Number of Predictions')
    ax6.set_title('Prediction Confidence Distribution')
    ax6.set_xticks(range(len(range_labels)))
    ax6.set_xticklabels(range_labels)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eval/plots/comprehensive_performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =================== SUMMARY STATISTICS ===================
    print(f"\n🎉 FINAL SUMMARY STATISTICS")
    print("="*50)
    
    # Overall performance at different IoU thresholds
    print(f"\nOverall Performance Summary:")
    for iou_thresh in [0.3, 0.5, 0.7]:
        acc, _, _ = handle_ties_and_calculate_accuracy(df, iou_threshold=iou_thresh)
        print(f"  Accuracy at IoU≥{iou_thresh}: {acc:.2f}%")
    
    print(f"\nRecall@K Summary:")
    for iou_thresh in [0.3, 0.5, 0.7]:
        print(f"  IoU≥{iou_thresh}:")
        recall_summary = calculate_recall_at_k(df, [1, 2, 3, 4, 5], iou_threshold=iou_thresh)
        for k, recall in recall_summary.items():
            print(f"    Recall@{k}: {recall:.2f}%")
    
    print(f"\nBounding Box Analysis Summary (Successful Predictions, IoU≥0.3):")
    if bbox_analysis:
        print(f"  Number of successful predictions analyzed: {len(bbox_analysis['successful_preds'])}")
        print(f"  Average width ratio (Pred/GT): {np.mean(bbox_analysis['width_ratios']):.3f}")
        print(f"  Average height ratio (Pred/GT): {np.mean(bbox_analysis['height_ratios']):.3f}")
        print(f"  Average area ratio (Pred/GT): {np.mean(bbox_analysis['area_ratios']):.3f}")
        print(f"  Predicted bbox avg width: {bbox_analysis['pred_width_stats']['mean']:.1f} px")
        print(f"  Ground truth bbox avg width: {bbox_analysis['gt_width_stats']['mean']:.1f} px")
        print(f"  Predicted bbox avg height: {bbox_analysis['pred_height_stats']['mean']:.1f} px")
        print(f"  Ground truth bbox avg height: {bbox_analysis['gt_height_stats']['mean']:.1f} px")
        
        # Identify trends
        width_diff = bbox_analysis['pred_width_stats']['mean'] - bbox_analysis['gt_width_stats']['mean']
        height_diff = bbox_analysis['pred_height_stats']['mean'] - bbox_analysis['gt_height_stats']['mean']
        area_diff = bbox_analysis['pred_area_stats']['mean'] - bbox_analysis['gt_area_stats']['mean']
        
        print(f"  Width difference (Pred - GT): {width_diff:+.1f} px")
        print(f"  Height difference (Pred - GT): {height_diff:+.1f} px")
        print(f"  Area difference (Pred - GT): {area_diff:+.0f} px²")
        
        if abs(width_diff) > 50:
            trend = "wider" if width_diff > 0 else "narrower"
            print(f"  📊 TREND: Predicted bboxes tend to be {trend} than ground truth")
        
        if abs(height_diff) > 50:
            trend = "taller" if height_diff > 0 else "shorter"
            print(f"  📊 TREND: Predicted bboxes tend to be {trend} than ground truth")
            
        if abs(area_diff) > 10000:
            trend = "larger" if area_diff > 0 else "smaller"
            print(f"  📊 TREND: Predicted bboxes tend to be {trend} than ground truth")
    
    print(f"\nSoft Unification Analysis Summary:")
    if successful_unif_analysis:
        print(f"  Total soft unifications (all): {len(all_unifications)}")
        print(f"  Total soft unifications (successful): {len(successful_unif_analysis['successful_unifications'])}")
        print(f"  Average probability (all): {np.mean(unif_probs):.6f}")
        print(f"  Average probability (successful): {np.mean(successful_unif_analysis['successful_unif_probs']):.6f}")
        
        prob_improvement = np.mean(successful_unif_analysis['successful_unif_probs']) - np.mean(unif_probs)
        print(f"  Probability improvement in successful predictions: {prob_improvement:+.6f}")
        
        print(f"  Top 5 unifications in successful predictions:")
        for unif, count in successful_unif_analysis['successful_unif_counter'].most_common(5):
            type_probs = [u[2] for u in successful_unif_analysis['successful_unifications'] if u[0] == unif]
            avg_prob = np.mean(type_probs) if type_probs else 0
            print(f"    {unif}: {count} times (avg prob: {avg_prob:.6f})")
    
    print(f"\nDataset Characteristics:")
    print(f"  Total queries: {total_queries}")
    print(f"  Average predictions per query: {len(df)/total_queries:.2f}")
    print(f"  Queries with probability ties: {len(tie_batches)}")
    if tie_batches:
        avg_tie_size = np.mean([t['num_tied'] for t in tie_batches])
        ties_with_correct = sum(1 for t in tie_batches if t['correct_in_tie'])
        print(f"  Average tie size: {avg_tie_size:.2f}")
        print(f"  Ties containing correct answer: {ties_with_correct}/{len(tie_batches)} ({ties_with_correct/len(tie_batches)*100:.1f}%)")
    
    print(f"  Unique object types (GT): {len(gt_objects)}")
    print(f"  Unique object types (predicted): {len(pred_objects)}")
    print(f"  Unique soft unification types: {len(unif_counter)}")
    
    print(f"\nQuery Complexity Breakdown:")
    if complexity_analysis:
        for complexity in sorted(complexity_analysis.keys()):
            stats = complexity_analysis[complexity]
            print(f"  {complexity} expressions: {stats['total_queries']} queries")
            print(f"    Object accuracy: {stats['correct_obj']/stats['total_queries']*100:.1f}%")
            print(f"    IoU≥0.3 accuracy: {stats['correct_iou_03']/stats['total_queries']*100:.1f}%")
            print(f"    IoU≥0.5 accuracy: {stats['correct_iou_05']/stats['total_queries']*100:.1f}%")
            print(f"    Avg probability: {stats['avg_prob']:.6f}")
            print(f"    Avg IoU: {stats['avg_iou']:.3f}")
    
    print(f"\nVisualization Files Generated:")
    visualizations = [
        'enhanced_iou_threshold_analysis.png',
        'bounding_box_analysis.png',
        'object_specific_bbox_analysis.png',
        'soft_unification_comparison.png',
        'comprehensive_performance_dashboard.png'
    ]
    
    for i, viz in enumerate(visualizations, 1):
        print(f"  {i}. {viz}")
    
    print(f"\n✅ Enhanced analysis complete! All plots saved to eval/plots/")
    print(f"📁 Generated {len(visualizations)} visualization files")
    
    return {
        'threshold_results': threshold_results,
        'tie_analysis': tie_batches,
        'complexity_analysis': complexity_analysis,
        'unification_stats': unif_counter,
        'bbox_analysis': bbox_analysis,
        'successful_unif_analysis': successful_unif_analysis,
        'total_queries': total_queries
    }

# Main execution
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('evaluation_metrics_sg_new.csv')
    
    # Run comprehensive analysis
    results = analyze_comprehensive_results(df)
    
    print(f"\n🎊 ANALYSIS COMPLETE!")
    print(f"Check eval/plots/ for all generated visualizations")