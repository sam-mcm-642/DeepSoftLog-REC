#!/usr/bin/env python3
"""
Main processor for converting referring expressions to Prolog-like triplets.
Optimized for Visual Genome embedding compatibility.
Outputs in sample_queries.json format.
"""

import anthropic
import os
import json
import time
import threading
import argparse
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict

class ClaudeReferringExpressionProcessor:
    def __init__(self, model_name="claude-3-5-sonnet-20241022", temperature=0.0):
        """Initialize the Claude processor."""
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        self.examples = self._get_visual_genome_examples()
    
    def _get_visual_genome_examples(self):
        """High-quality examples using Visual Genome standard predicates."""
        return [
            "INPUT: The person sitting on a bench next to another person who is on a skateboard.\nOUTPUT: target(X), type(X, person), expression(on, X, bench), expression(nextTo, X, person), expression(on, person, skateboard)",
            
            "INPUT: The white plate that is to the right of the tomato that is in the bowl.\nOUTPUT: target(X), type(X, plate), expression(hasAttribute, X, white), expression(rightOf, X, tomato), expression(in, tomato, bowl)",
            
            "INPUT: The red car parked beside a tree in front of a house.\nOUTPUT: target(X), type(X, car), expression(hasAttribute, X, red), expression(nextTo, X, tree), expression(inFrontOf, X, building)",
            
            "INPUT: The man wearing a blue shirt standing behind a woman.\nOUTPUT: target(X), type(X, person), expression(wearing, X, shirt), expression(hasAttribute, shirt, blue), expression(behind, X, person)",
            
            "INPUT: The dog lying under the table near a chair.\nOUTPUT: target(X), type(X, dog), expression(under, X, table), expression(near, X, chair)",
            
            "INPUT: The cat sitting on the windowsill looking at a bird.\nOUTPUT: target(X), type(X, cat), expression(on, X, window), expression(lookingAt, X, bird)",
            
            "INPUT: The striped zebra that is in front of the gray sand and walking by the white zebra.\nOUTPUT: target(X), type(X, zebra), expression(hasAttribute, X, striped), expression(inFrontOf, X, ground), expression(hasAttribute, ground, gray), expression(near, X, zebra), expression(hasAttribute, zebra, white)",
            
            "INPUT: The person holding a red umbrella while walking on the street.\nOUTPUT: target(X), type(X, person), expression(holding, X, umbrella), expression(hasAttribute, umbrella, red), expression(on, X, street)",
            
            "INPUT: plate on the left that is white and to the left of food.\nOUTPUT: target(X), type(X, plate), expression(hasPosition, X, left), expression(hasAttribute, X, white), expression(leftOf, X, food)",
            
            "INPUT: the first rock from the left that is large and to the left of gray pavement.\nOUTPUT: target(X), type(X, rock), expression(hasAttribute, X, large), expression(leftOf, X, pavement), expression(hasAttribute, pavement, gray)"
        ]
    
    def process_expression(self, expression_text):
        """Process a single referring expression."""
        prompt = f"""You are an expert in converting referring expressions to logical facts. Use simple, consistent predicate names that match Visual Genome relationships.

RULES:
1. Use simple predicate names: on, in, under, above, nextTo, behind, inFrontOf, leftOf, rightOf, near, wearing, holding, lookingAt, etc.
2. Use camelCase for multi-word predicates and objects: nextTo, inFrontOf,
lookingAt, hasAttribute, hasPosition, fromDirection, trashCan, etc.
3. Use "hasAttribute" for colors, sizes, and properties: expression(hasAttribute, X, red)
4. Output format: target(X), type(X, category), expression(predicate, subject, object), ...
5. Consistently use X as the variable for the main object, and do not include
any other variables.
6. IGNORE ordinal positions and directional references (first, second, from the
left/right, etc.) - only use the remaining descriptive attributes and spatial
relations

CRITICAL: Output ONLY comma-separated facts on one line, nothing else.

Examples:

{chr(10).join(self.examples)}

##Your Task##

INPUT: {expression_text}
OUTPUT: """
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=600,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")

class TokenBucket:
    """Rate limiting using token bucket algorithm."""
    def __init__(self, tokens, refill_rate):
        self.tokens = tokens
        self.capacity = tokens
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def take(self, count=1):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens += elapsed * self.refill_rate
            self.tokens = min(self.tokens, self.capacity)
            self.last_refill = now

            if self.tokens >= count:
                self.tokens -= count
                return True
            return False

    def wait_for_token(self, count=1):
        while not self.take(count):
            sleep_time = max(1, (count - self.tokens) / self.refill_rate)
            time.sleep(sleep_time)

def clean_response(response_text):
    """Extract the clean Prolog facts from Claude's response."""
    lines = response_text.strip().split('\n')
    
    # Find line with target(X)
    for line in lines:
        if 'target(X)' in line:
            return line.strip()
    
    # Fallback to first non-empty line that looks like output
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('INPUT:', 'OUTPUT:', '#', 'Here', 'The')):
            return line
    
    return response_text.strip()

def validate_output(output_str):
    """Validate the output format."""
    if not output_str or 'target(X)' not in output_str:
        return False
    
    if 'type(X,' not in output_str:
        return False
    
    if 'expression(' not in output_str:
        return False
    
    return True

def load_cops_ref_data(file_path):
    """Load COPS-Ref format data with additional fields."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for ref in data.get('refs', []):
        if ref.get('sentences'):
            for sentence in ref['sentences']:
                if sentence.get('sent'):
                    processed_item = {
                        'id': f"{ref.get('image_id', 'unknown')}_{sentence.get('sent_id', 'unknown')}",
                        'input': sentence.get('sent', '').strip(),  # Don't lowercase for better processing
                        'image_id': ref.get('image_id'),
                        'sent_id': sentence.get('sent_id'),
                        'expression': sentence.get('sent', ''),  # Original expression
                        'object': ref.get('name', ''),           # Ground truth object
                        'object_id': ref.get('objectId', ''),    # Object ID
                        'bbox': ref.get('box', [])               # Bounding box [x,y,w,h]
                    }
                    processed_data.append(processed_item)
    
    return processed_data

def load_jsonl_data(file_path):
    """Load JSONL format data."""
    processed_data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                image = json.loads(line)
                file_name = "_".join(image["file_name"].split("_")[:-1]) + ".jpg"
                sentences = image.get("sentences", [])
                for s in sentences:
                    sent_id = s["sent_id"]
                    sentence = s["raw"].strip()
                    processed_item = {
                        'id': f"{file_name}_{sent_id}",
                        'input': sentence,
                        'file_name': file_name,
                        'sent_id': sent_id,
                        'image_id': file_name.split('.')[0],  # Extract image_id from filename
                        'expression': sentence,
                        'object': 'unknown',  # JSONL format may not have ground truth
                        'bbox': []
                    }
                    processed_data.append(processed_item)
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line: {line.strip()}")
                continue
    
    return processed_data

def process_single_item(primary_agent, fallback_agent, item, results, lock):
    """Process a single referring expression and store result."""
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            # Process with Claude
            if attempt < 2:
                response = primary_agent.process_expression(item['input'])
            else:
                response = fallback_agent.process_expression(item['input'])
            
            cleaned_output = clean_response(response)
            
            if validate_output(cleaned_output):
                # Create result in sample_queries.json format
                result = {
                    'image_id': item['image_id'],
                    'query': cleaned_output,
                    'target': [
                        item['object'],
                        item['bbox']
                    ],
                    'probability': 1.0  # Default probability for generated queries
                }
                
                with lock:
                    results.append(result)
                return
            else:
                raise ValueError("Invalid output format")
                
        except Exception as e:
            if attempt == max_attempts - 1:
                # Final fallback
                fallback_output = f"target(X), type(X, {item['object'] if item['object'] else 'object'}), expression(hasAttribute, X, unknown)"
                result = {
                    'image_id': item['image_id'],
                    'query': fallback_output,
                    'target': [
                        item['object'] if item['object'] else 'unknown',
                        item['bbox']
                    ],
                    'probability': 1.0
                }
                
                with lock:
                    results.append(result)
                
                logging.warning(f"Used fallback for: {item['input']}")
            else:
                time.sleep(1)

def threaded_processor(primary_agent, fallback_agent, item, results, lock, semaphore, bucket):
    """Thread wrapper for processing with rate limiting."""
    estimated_tokens = 800
    bucket.wait_for_token(estimated_tokens)
    
    try:
        process_single_item(primary_agent, fallback_agent, item, results, lock)
    finally:
        semaphore.release()

def save_results(results, output_file):
    """Save results in sample_queries.json format."""
    # Group by image_id and sort
    output_data = {
        "queries": sorted(results, key=lambda x: str(x['image_id']))
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved {len(results)} queries to {output_file}")
    
    # Print summary statistics
    image_count = len(set(item['image_id'] for item in results))
    logging.info(f"Processed {image_count} unique images")
    
    # Count by target object
    object_counts = defaultdict(int)
    for item in results:
        object_counts[item['target'][0]] += 1
    
    logging.info("Top target objects:")
    for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        logging.info(f"  {obj}: {count}")

def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Process referring expressions to Prolog triplets')
    
    # Define default file paths - CHANGE THESE TO YOUR PATHS
    default_input = 'data/cops_ref_test_sample_1000.json'
    default_output = 'data/final_queries.json'
    
    parser.add_argument('input_file', nargs='?', default=default_input, 
                       help=f'Input JSON or JSONL file (default: {default_input})')
    parser.add_argument('output_file', nargs='?', default=default_output,
                       help=f'Output JSON file (default: {default_output})')
    parser.add_argument('--max_threads', type=int, default=10, help='Maximum number of threads')
    parser.add_argument('--model', default='claude-3-5-sonnet-20241022', help='Claude model to use')
    parser.add_argument('--fallback_model', default='claude-3-5-haiku-20241022', help='Fallback model')
    parser.add_argument('--limit', type=int, help='Limit number of expressions to process (for testing)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('processing.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load environment
    load_dotenv()
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found! Add it to your .env file.")
    
    # Initialize agents
    logging.info("Initializing Claude agents...")
    primary_agent = ClaudeReferringExpressionProcessor(model_name=args.model)
    fallback_agent = ClaudeReferringExpressionProcessor(model_name=args.fallback_model)
    
    # Load data
    logging.info(f"Loading data from {args.input_file}...")
    try:
        if args.input_file.endswith('.json'):
            data = load_cops_ref_data(args.input_file)
        else:
            data = load_jsonl_data(args.input_file)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return
    
    if not data:
        logging.error("No data found in input file!")
        return
    
    # Apply limit if specified
    if args.limit:
        data = data[:args.limit]
        logging.info(f"Limited to {len(data)} expressions for testing")
    
    logging.info(f"Loaded {len(data)} referring expressions")
    
    # Setup threading and rate limiting
    lock = threading.Lock()
    semaphore = threading.Semaphore(args.max_threads)
    bucket = TokenBucket(tokens=50000, refill_rate=1000)  # Claude rate limits
    results = []
    
    # Process data
    logging.info("Starting processing...")
    threads = []
    
    for item in tqdm(data, desc="Processing expressions"):
        semaphore.acquire()
        thread = threading.Thread(
            target=threaded_processor,
            args=(primary_agent, fallback_agent, item, results, lock, semaphore, bucket)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Save results
    if results:
        save_results(results, args.output_file)
        logging.info(f"Processing complete! Results saved to {args.output_file}")
    else:
        logging.error("No results generated!")

if __name__ == "__main__":
    main()