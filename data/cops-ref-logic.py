#!/usr/bin/env python3
"""
Analyze COPS-Ref dataset for logic types and negation patterns.
"""

import json
from collections import defaultdict
import re

def analyze_dataset_logic(file_path):
    """Analyze logic types and negation patterns in COPS-Ref dataset."""
    
    # Load the dataset
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Track logic types
    logic_counts = defaultdict(int)
    
    # Track negation patterns
    negation_examples = []
    negation_patterns = [
        r'\bnot\b',
        r'\bisn\'t\b', 
        r'\bare not\b',
        r'\bis not\b',
        r'\bno\b',
        r'\bwithout\b'
    ]
    
    # Track all expressions for analysis
    all_expressions = []
    
    print("Analyzing dataset...")
    
    for ref in data.get('refs', []):
        if not ref or not ref.get('sentences'):
            continue
            
        logic_type = ref.get('logic')
        if logic_type:
            logic_counts[logic_type] += 1
        
        # Check sentences for negation patterns
        for sentence in ref['sentences']:
            expr = sentence.get('sent', '')
            if expr:
                all_expressions.append({
                    'expression': expr,
                    'logic': logic_type,
                    'ref_id': ref.get('ref_id'),
                    'image_id': ref.get('image_id')
                })
                
                # Check for negation patterns
                expr_lower = expr.lower()
                for pattern in negation_patterns:
                    if re.search(pattern, expr_lower):
                        negation_examples.append({
                            'expression': expr,
                            'logic': logic_type,
                            'pattern': pattern,
                            'ref_id': ref.get('ref_id')
                        })
                        break
    
    # Print results
    print("\n" + "="*60)
    print("LOGIC TYPE DISTRIBUTION:")
    print("="*60)
    
    for logic_type, count in sorted(logic_counts.items()):
        print(f"{logic_type:15s}: {count:6d} cases")
    
    print(f"\nTotal expressions: {len(all_expressions)}")
    
    print("\n" + "="*60)
    print("NEGATION ANALYSIS:")
    print("="*60)
    
    if negation_examples:
        print(f"Found {len(negation_examples)} expressions with negation patterns:")
        print()
        
        # Group by logic type
        negation_by_logic = defaultdict(list)
        for neg in negation_examples:
            negation_by_logic[neg['logic']].append(neg)
        
        for logic_type, examples in negation_by_logic.items():
            print(f"Logic type '{logic_type}': {len(examples)} negation cases")
            for example in examples[:3]:  # Show first 3 examples
                print(f"  - {example['expression']}")
            if len(examples) > 3:
                print(f"  ... and {len(examples) - 3} more")
            print()
    else:
        print("No negation patterns found!")
    
    print("="*60)
    print("SPECIFIC EXAMPLES:")
    print("="*60)
    
    # Look for the specific "not orange" case
    orange_case = None
    for expr_data in all_expressions:
        if 'not orange' in expr_data['expression'].lower():
            orange_case = expr_data
            break
    
    if orange_case:
        print(f"Found 'not orange' case:")
        print(f"  Expression: {orange_case['expression']}")
        print(f"  Logic type: {orange_case['logic']}")
        print(f"  Ref ID: {orange_case['ref_id']}")
    else:
        print("'not orange' case not found in this dataset")
    
    # Show a few examples of each logic type
    print(f"\nSample expressions by logic type:")
    for logic_type in sorted(logic_counts.keys()):
        examples = [e for e in all_expressions if e['logic'] == logic_type]
        if examples:
            print(f"\n{logic_type}:")
            for i, example in enumerate(examples[:2]):  # Show 2 examples
                print(f"  {i+1}. {example['expression']}")
    
    return {
        'logic_counts': dict(logic_counts),
        'negation_examples': negation_examples,
        'total_expressions': len(all_expressions)
    }

def check_for_explicit_not_logic(file_path):
    """Specifically check if 'not' is used as a logic type."""
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    not_logic_cases = []
    
    for ref in data.get('refs', []):
        if ref and ref.get('logic') == 'not':
            for sentence in ref.get('sentences', []):
                not_logic_cases.append({
                    'expression': sentence.get('sent', ''),
                    'ref_id': ref.get('ref_id')
                })
    
    print(f"Explicit 'not' logic cases: {len(not_logic_cases)}")
    for case in not_logic_cases:
        print(f"  - {case['expression']}")
    
    return not_logic_cases

if __name__ == "__main__":
    # Replace with your dataset file path
    dataset_file = "data/cops_ref_test_sample_1000.json"  # or your full dataset
    
    print("Checking for explicit 'not' logic type:")
    explicit_not = check_for_explicit_not_logic(dataset_file)
    
    print("\n" + "="*60)
    print("FULL ANALYSIS:")
    
    results = analyze_dataset_logic(dataset_file)
    
    print(f"\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Total logic types found: {len(results['logic_counts'])}")
    print(f"Expressions with negation patterns: {len(results['negation_examples'])}")
    print(f"Explicit 'not' logic cases: {len(explicit_not)}")
    
    # Recommendation
    print(f"\n" + "="*60)
    print("FILTERING RECOMMENDATION:")
    print("="*60)
    
    if 'or' in results['logic_counts']:
        print(f"❌ Found {results['logic_counts']['or']} OR cases - recommend filtering")
    
    if explicit_not:
        print(f"❌ Found {len(explicit_not)} explicit NOT logic cases - recommend filtering")
    elif results['negation_examples']:
        print(f"⚠️  Found {len(results['negation_examples'])} text-based negation cases")
        print("   Consider filtering or handling in prompt")
    else:
        print("✅ No negation cases found")
    
    safe_logic_types = [k for k in results['logic_counts'].keys() 
                       if k not in ['or', 'not']]
    safe_count = sum(results['logic_counts'][k] for k in safe_logic_types)
    print(f"✅ {safe_count} expressions with safe logic types: {safe_logic_types}")