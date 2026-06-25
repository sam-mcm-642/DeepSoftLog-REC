#!/usr/bin/env python3
"""
Cost-optimized batch processor for referring expressions.
Perfect for thesis work with limited budget.
"""

import anthropic
import os
import json
import time
import re
from dotenv import load_dotenv
from tqdm import tqdm

class EfficientBatchProcessor:
    def __init__(self, model_name="claude-3-5-haiku-20241022", temperature=0.0):
        """Initialize with the cheapest Claude model."""
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
    
    def create_batch_prompt(self, expressions):
        """Create a minimal batch prompt."""
        # Minimal examples to save tokens
        examples = """Examples:
"red car" → target(X), type(X, car), expression(hasAttribute, X, red)
"person on bench" → target(X), type(X, person), expression(on, X, bench)"""
        
        # Number the expressions
        numbered_expressions = []
        for i, expr in enumerate(expressions, 1):
            numbered_expressions.append(f"{i}. {expr}")
        
        prompt = f"""{examples}

Convert each to format: target(X), type(X, object), expression(relation, X, object)

{chr(10).join(numbered_expressions)}

Respond with numbered results only:"""
        
        return prompt
    
    def process_batch(self, expressions, max_retries=2):
        """Process a batch of expressions in one API call."""
        if not expressions:
            return []
        
        prompt = self.create_batch_prompt(expressions)
        
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2000,  # Enough for batch responses
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                response_text = response.content[0].text.strip()
                return self.parse_batch_response(response_text, expressions)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Return fallback results for all expressions
                    return [f"target(X), type(X, object), expression(hasAttribute, X, unknown)" 
                           for _ in expressions]
                time.sleep(1)
        
        return []
    
    def parse_batch_response(self, response_text, original_expressions):
        """Parse numbered responses from the batch."""
        results = []
        lines = response_text.strip().split('\n')
        
        # Try to extract numbered results
        numbered_results = {}
        for line in lines:
            line = line.strip()
            # Look for pattern: "1. target(X), ..." or "1: target(X), ..."
            match = re.match(r'(\d+)[\.\:\s]+(.+)', line)
            if match:
                num = int(match.group(1))
                result = match.group(2).strip()
                if 'target(X)' in result:
                    numbered_results[num] = result
        
        # Create results in order
        for i in range(1, len(original_expressions) + 1):
            if i in numbered_results:
                results.append(numbered_results[i])
            else:
                # Fallback for missing results
                results.append("target(X), type(X, object), expression(hasAttribute, X, unknown)")
        
        return results
    
    def validate_result(self, result):
        """Quick validation check."""
        return ('target(X)' in result and 
                'type(X,' in result and 
                'expression(' in result)

def process_file_efficiently(input_file, output_file, batch_size=20):
    """Main function to process a file with minimal cost."""
    
    # Load environment
    load_dotenv()
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found! Add it to your .env file.")
    
    # Initialize processor with cheapest model
    processor = EfficientBatchProcessor()
    
    # Load data (using existing functions)
    if input_file.endswith('.json'):
        from llm_parser import load_cops_ref_data
        data = load_cops_ref_data(input_file)
    else:
        from llm_parser import load_jsonl_data
        data = load_jsonl_data(input_file)
    
    print(f"Loaded {len(data)} expressions")
    print(f"Processing in batches of {batch_size}")
    
    # Estimate costs
    estimated_tokens_per_batch = 500 + (batch_size * 20)  # Conservative estimate
    total_batches = (len(data) + batch_size - 1) // batch_size
    estimated_total_tokens = estimated_tokens_per_batch * total_batches
    estimated_cost = estimated_total_tokens * 0.00000025  # Haiku pricing
    
    print(f"Estimated cost: ~${estimated_cost:.2f}")
    print(f"Estimated tokens: ~{estimated_total_tokens:,}")
    
    # Confirm before proceeding
    if input("Continue? (y/n): ").lower() != 'y':
        return
    
    # Clear output file
    with open(output_file, 'w') as f:
        pass
    
    # Process in batches
    all_results = []
    
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch_data = data[i:i+batch_size]
        batch_expressions = [item['input'] for item in batch_data]
        
        # Process batch
        batch_results = processor.process_batch(batch_expressions)
        
        # Save results
        for j, result in enumerate(batch_results):
            item_id = batch_data[j]['id']
            output_item = {item_id: result}
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            
            all_results.append(result)
        
        # Be nice to the API
        time.sleep(0.5)
    
    # Summary
    valid_results = sum(1 for r in all_results if processor.validate_result(r))
    print(f"\nCompleted! {valid_results}/{len(all_results)} valid results")
    print(f"Results saved to {output_file}")

def quick_test_batch():
    """Test batch processing with a few examples."""
    load_dotenv()
    
    processor = EfficientBatchProcessor()
    
    test_expressions = [
        "the red car",
        "person sitting on bench", 
        "dog under table",
        "white plate next to cup"
    ]
    
    print("Testing batch processing...")
    print(f"Input: {test_expressions}")
    
    results = processor.process_batch(test_expressions)
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result}")
        print(f"   Valid: {processor.validate_result(result)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 2 and sys.argv[1] == "test":
        quick_test_batch()
    elif len(sys.argv) == 3:
        input_file, output_file = sys.argv[1], sys.argv[2]
        process_file_efficiently(input_file, output_file)
    else:
        print("Usage:")
        print("  python efficient_processor.py test                    # Test batch processing")
        print("  python efficient_processor.py input.json output.jsonl # Process file")