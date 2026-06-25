import json
import random
from pathlib import Path
from collections import Counter

def has_ordinals(expression):
    """Check if expression contains ordinal patterns that scene graphs cannot prove"""
    ordinal_patterns = ['first', 'second', 'third', 'from the left', 'from the right']
    return any(pattern in expression.lower() for pattern in ordinal_patterns)

def extract_filtered_test_sample(input_path, output_path, sample_size=1000, random_seed=42, 
                                excluded_logic=['not', 'or', 'same'], filter_ordinals=True):
    """
    Extract a random sample of test instances from Cops-Ref dataset, excluding certain logic types and ordinals
    
    Args:
        input_path: Path to the original Cops-Ref JSON file
        output_path: Path where to save the sampled test data
        sample_size: Number of test instances to sample (default: 1000)
        random_seed: Random seed for reproducibility (default: 42)
        excluded_logic: List of logic types to exclude (default: ['not', 'or', 'same'])
        filter_ordinals: Whether to filter out expressions with ordinals (default: True)
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    print(f"🔄 Loading data from {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict) and 'refs' in data:
        refs = data['refs']
        maintain_wrapper = True
    elif isinstance(data, list):
        refs = data
        maintain_wrapper = False
    else:
        raise ValueError("Unexpected JSON structure")
    
    # Filter for test instances only
    print("🔍 Filtering test instances...")
    all_test_instances = []
    for ref in refs:
        if ref is not None and isinstance(ref, dict) and ref.get('split') == 'test':
            all_test_instances.append(ref)
    
    print(f"📊 Found {len(all_test_instances)} total test instances")
    
    # Apply logic and ordinal filtering
    print(f"🚫 Excluding logic types: {excluded_logic}")
    if filter_ordinals:
        print(f"🚫 Excluding expressions with ordinals: first, second, third, 'from the left/right'")
    
    filtered_test_instances = []
    excluded_by_logic = []
    excluded_by_ordinals = []
    
    # Count all logic types in test set
    all_logic_counts = Counter()
    ordinal_count = 0
    
    for ref in all_test_instances:
        logic_type = ref.get('logic')
        all_logic_counts[logic_type] += 1
        
        # Check if excluded by logic type
        if logic_type in excluded_logic:
            excluded_by_logic.append(ref)
            continue
        
        # Check if excluded by ordinals (only if logic type is safe)
        has_ordinal_expression = False
        if filter_ordinals and ref.get('sentences'):
            for sentence in ref['sentences']:
                if sentence.get('sent') and has_ordinals(sentence['sent']):
                    has_ordinal_expression = True
                    ordinal_count += 1
                    break
        
        if has_ordinal_expression:
            excluded_by_ordinals.append(ref)
            continue
        
        # If we get here, the instance passes all filters
        filtered_test_instances.append(ref)
    
    print(f"\n📈 LOGIC TYPE DISTRIBUTION IN TEST SET:")
    print("-" * 40)
    for logic_type, count in sorted(all_logic_counts.items()):
        status = "❌ EXCLUDED" if logic_type in excluded_logic else "✅ INCLUDED"
        percentage = (count / len(all_test_instances)) * 100
        print(f"  {logic_type}: {count} ({percentage:.1f}%) {status}")
    
    print(f"\n📊 FILTERING RESULTS:")
    print("-" * 40)
    print(f"  Original test instances: {len(all_test_instances)}")
    print(f"  Excluded by logic type: {len(excluded_by_logic)}")
    if filter_ordinals:
        print(f"  Excluded by ordinals: {len(excluded_by_ordinals)}")
    print(f"  Final filtered instances: {len(filtered_test_instances)}")
    
    total_excluded = len(excluded_by_logic) + len(excluded_by_ordinals)
    exclusion_rate = (total_excluded / len(all_test_instances)) * 100
    print(f"  Total exclusion rate: {exclusion_rate:.1f}%")
    
    # Check if we have enough filtered test instances
    if len(filtered_test_instances) < sample_size:
        print(f"⚠️  Warning: Only {len(filtered_test_instances)} filtered instances available")
        print(f"   Adjusting sample size to {len(filtered_test_instances)}")
        sample_size = len(filtered_test_instances)
        sampled_instances = filtered_test_instances
    else:
        # Randomly sample the requested number
        print(f"🎲 Randomly sampling {sample_size} instances from filtered set...")
        sampled_instances = random.sample(filtered_test_instances, sample_size)
    
    # Prepare output data in same format as input
    if maintain_wrapper:
        output_data = {"refs": sampled_instances}
    else:
        output_data = sampled_instances
    
    # Save to output file
    print(f"💾 Saving sample to {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print(f"\n✅ SUCCESS!")
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    print(f"📊 Sampled: {len(sampled_instances)} filtered test instances")
    print(f"🎯 Random seed: {random_seed}")
    print(f"🚫 Excluded logic types: {excluded_logic}")
    if filter_ordinals:
        print(f"🚫 Filtered ordinal expressions: {len(excluded_by_ordinals)}")
    
    # Show statistics about the filtered sample
    print(f"\n📈 FILTERED SAMPLE STATISTICS:")
    print("-" * 40)
    
    # Count logic types in sample
    sample_logic_counts = Counter()
    for instance in sampled_instances:
        logic = instance.get('logic', 'unknown')
        sample_logic_counts[logic] += 1
    
    print("Logic types in filtered sample:")
    for logic, count in sorted(sample_logic_counts.items()):
        percentage = (count / len(sampled_instances)) * 100
        print(f"  {logic}: {count} ({percentage:.1f}%)")
    
    # Count unique images
    unique_images = set()
    for instance in sampled_instances:
        if 'imageId' in instance:
            unique_images.add(instance['imageId'])
    
    print(f"\nUnique images: {len(unique_images)}")
    print(f"Avg expressions per image: {len(sampled_instances) / len(unique_images):.1f}")
    
    # Show what percentage of original test set this represents
    original_test_count = len(all_test_instances)
    coverage = (len(sampled_instances) / original_test_count) * 100
    print(f"Coverage of original test set: {coverage:.1f}%")
    
    return output_path, len(sampled_instances), len(excluded_by_logic) + len(excluded_by_ordinals)

def analyze_available_logic_types(input_path, excluded_logic=['not', 'or', 'same'], filter_ordinals=True):
    """
    Analyze what logic types are available in the test set before filtering
    """
    print(f"🔍 ANALYZING AVAILABLE LOGIC TYPES")
    print("=" * 50)
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'refs' in data:
        refs = data['refs']
    else:
        refs = data
    
    # Get test instances
    test_instances = [ref for ref in refs 
                     if ref is not None and isinstance(ref, dict) and ref.get('split') == 'test']
    
    # Count logic types
    logic_counts = Counter()
    ordinal_expressions = 0
    
    for ref in test_instances:
        logic_type = ref.get('logic')
        logic_counts[logic_type] += 1
        
        # Count ordinal expressions
        if filter_ordinals and ref.get('sentences'):
            for sentence in ref['sentences']:
                if sentence.get('sent') and has_ordinals(sentence['sent']):
                    ordinal_expressions += 1
                    break
    
    print(f"Total test instances: {len(test_instances)}")
    print(f"\nLogic type distribution:")
    
    for logic_type, count in sorted(logic_counts.items()):
        percentage = (count / len(test_instances)) * 100
        status = "❌ EXCLUDED" if logic_type in excluded_logic else "✅ INCLUDED"
        print(f"  {logic_type}: {count} ({percentage:.1f}%) {status}")
    
    # Calculate what would remain after filtering
    excluded_by_logic = sum(logic_counts.get(logic, 0) for logic in excluded_logic)
    remaining_after_logic = len(test_instances) - excluded_by_logic
    
    if filter_ordinals:
        print(f"\nOrdinal expressions found: {ordinal_expressions}")
        # Note: Some ordinal expressions might also be excluded by logic, so this is an estimate
        estimated_remaining = remaining_after_logic - ordinal_expressions
        if estimated_remaining < 0:
            estimated_remaining = 0
        
        remaining_percentage = (estimated_remaining / len(test_instances)) * 100
        print(f"\nAfter excluding logic types {excluded_logic} and ordinals:")
        print(f"  Estimated remaining: ~{estimated_remaining} (~{remaining_percentage:.1f}%)")
    else:
        remaining_percentage = (remaining_after_logic / len(test_instances)) * 100
        print(f"\nAfter excluding logic types {excluded_logic}:")
        print(f"  Remaining: {remaining_after_logic} ({remaining_percentage:.1f}%)")
    
    return remaining_after_logic if not filter_ordinals else estimated_remaining

def show_ordinal_examples(input_path, num_examples=5):
    """Show examples of expressions that would be filtered due to ordinals"""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'refs' in data:
        refs = data['refs']
    else:
        refs = data
    
    print(f"\n🔍 EXAMPLES OF ORDINAL EXPRESSIONS (to be filtered):")
    print("-" * 60)
    
    ordinal_examples = []
    for ref in refs:
        if ref and ref.get('sentences') and ref.get('split') == 'test':
            for sentence in ref['sentences']:
                if sentence.get('sent') and has_ordinals(sentence['sent']):
                    ordinal_examples.append(sentence['sent'])
                    if len(ordinal_examples) >= num_examples:
                        break
        if len(ordinal_examples) >= num_examples:
            break
    
    for i, example in enumerate(ordinal_examples, 1):
        print(f"  {i}. {example}")
    
    if ordinal_examples:
        print(f"\n✅ These will be filtered out to keep only provable expressions")

def main():
    # Configuration
    input_file = "data/data.json"  # Update this path
    output_file = "data/cops_ref_test_sample_1000.json"
    sample_size = 1000
    random_seed = 42
    excluded_logic = ['not', 'or', 'same']  # Logic types to exclude
    filter_ordinals = True  # Whether to filter ordinal expressions
    
    try:
        # Show examples of what will be filtered
        show_ordinal_examples(input_file)
        
        # First analyze what's available
        available_count = analyze_available_logic_types(input_file, excluded_logic, filter_ordinals)
        
        if available_count < sample_size:
            print(f"\n⚠️  Warning: Only ~{available_count} instances available after filtering")
            print(f"   Consider reducing sample_size to {available_count} or less")
        
        print(f"\n" + "=" * 50)
        
        # Extract filtered sample
        output_path, sampled_count, excluded_count = extract_filtered_test_sample(
            input_path=input_file,
            output_path=output_file,
            sample_size=sample_size,
            random_seed=random_seed,
            excluded_logic=excluded_logic,
            filter_ordinals=filter_ordinals
        )
        
        print(f"\n🎉 FILTERED SAMPLE READY!")
        print(f"📝 Load it with:")
        print(f"```python")
        print(f"import json")
        print(f"with open('{output_file}', 'r') as f:")
        print(f"    test_data = json.load(f)")
        print(f"```")
        
        print(f"\n✨ Your model will now only see:")
        print(f"   ✅ chain, and, unary logic types")
        print(f"   🚫 No 'not', 'or', or 'same' expressions")
        if filter_ordinals:
            print(f"   🚫 No ordinal expressions (first, second, from left/right)")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find input file '{input_file}'")
        print("Please update the input_file path in the script")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()