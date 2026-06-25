#!/usr/bin/env python3
"""
Comparison script for Sonnet vs Haiku referring expression parsers.
Processes COPS-Ref sample data and generates detailed comparison output.
"""

import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Add the source directory to Python path
sys.path.append('src/models/text/triplets_chatgpt')

from llm_parser import ClaudeReferringExpressionProcessor, clean_response, validate_output
from llm_parser_efficient import EfficientBatchProcessor

class ModelComparator:
    def __init__(self):
        """Initialize both models and cost tracking."""
        load_dotenv()
        
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found! Add it to your .env file.")
        
        # Initialize models
        self.sonnet_processor = ClaudeReferringExpressionProcessor(
            model_name="claude-3-5-sonnet-20241022",
            temperature=0.0
        )
        
        self.haiku_processor = EfficientBatchProcessor(
            model_name="claude-3-5-haiku-20241022", 
            temperature=0.0
        )
        
        # Cost tracking (approximate tokens per request)
        self.sonnet_cost_per_1k_tokens = 0.003  # $3 per 1M input tokens
        self.haiku_cost_per_1k_tokens = 0.00025  # $0.25 per 1M input tokens
        
        self.results = []
        self.total_sonnet_tokens = 0
        self.total_haiku_tokens = 0
    
    def load_cops_ref_data(self, file_path):
        """Load and extract expressions from COPS-Ref format."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        expressions = []
        for ref in data.get('refs', []):
            if ref.get('sentences'):
                for sentence in ref['sentences']:
                    if sentence.get('sent'):
                        expressions.append({
                            'id': f"{ref.get('image_id', 'unknown')}_{sentence.get('sent_id', 'unknown')}",
                            'expression': sentence.get('sent', '').strip(),
                            'image_id': ref.get('image_id'),
                            'sent_id': sentence.get('sent_id'),
                            'object_name': ref.get('name', 'unknown'),
                            'category_id': ref.get('category_id', 'unknown'),
                            'attributes': ref.get('attributes', [])
                        })
        
        return expressions
    
    def estimate_tokens(self, text):
        """Rough token estimation (1 token ≈ 4 characters)."""
        return len(text) // 4 + 50  # Add overhead for prompt
    
    def process_with_sonnet(self, expression):
        """Process single expression with Sonnet."""
        start_time = time.time()
        
        try:
            raw_response = self.sonnet_processor.process_expression(expression)
            cleaned_response = clean_response(raw_response)
            
            # Estimate tokens (input + output)
            estimated_tokens = self.estimate_tokens(expression) + self.estimate_tokens(cleaned_response)
            self.total_sonnet_tokens += estimated_tokens
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'output': cleaned_response,
                'raw_output': raw_response,
                'valid': validate_output(cleaned_response),
                'estimated_tokens': estimated_tokens,
                'processing_time': processing_time,
                'error': None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            estimated_tokens = self.estimate_tokens(expression)
            self.total_sonnet_tokens += estimated_tokens
            
            return {
                'success': False,
                'output': None,
                'raw_output': None,
                'valid': False,
                'estimated_tokens': estimated_tokens,
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def process_with_haiku_single(self, expression):
        """Process single expression with Haiku (using batch size 1)."""
        start_time = time.time()
        
        try:
            batch_results = self.haiku_processor.process_batch([expression])
            result = batch_results[0] if batch_results else None
            
            # Estimate tokens
            estimated_tokens = self.estimate_tokens(expression) + self.estimate_tokens(result or "")
            self.total_haiku_tokens += estimated_tokens
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'output': result,
                'raw_output': result,  # Haiku processor already cleans
                'valid': self.haiku_processor.validate_result(result) if result else False,
                'estimated_tokens': estimated_tokens,
                'processing_time': processing_time,
                'error': None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            estimated_tokens = self.estimate_tokens(expression)
            self.total_haiku_tokens += estimated_tokens
            
            return {
                'success': False,
                'output': None,
                'raw_output': None,
                'valid': False,
                'estimated_tokens': estimated_tokens,
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def compare_models(self, expressions):
        """Process all expressions with both models."""
        print(f"Processing {len(expressions)} expressions with both models...")
        print("=" * 60)
        
        for i, expr_data in enumerate(expressions, 1):
            expression = expr_data['expression']
            print(f"Processing {i}/{len(expressions)}: {expression[:50]}...")
            
            # Process with Sonnet
            print("  → Sonnet...", end="")
            sonnet_result = self.process_with_sonnet(expression)
            print(f" {'✓' if sonnet_result['success'] else '✗'}")
            
            # Small delay between models
            time.sleep(0.5)
            
            # Process with Haiku  
            print("  → Haiku...", end="")
            haiku_result = self.process_with_haiku_single(expression)
            print(f" {'✓' if haiku_result['success'] else '✗'}")
            
            # Store results
            result_entry = {
                # Original data
                'id': expr_data['id'],
                'expression': expression,
                'image_id': expr_data['image_id'],
                'sent_id': expr_data['sent_id'],
                'object_name': expr_data['object_name'],
                'category_id': expr_data['category_id'],
                'attributes': expr_data['attributes'],
                
                # Sonnet results
                'sonnet_success': sonnet_result['success'],
                'sonnet_output': sonnet_result['output'],
                'sonnet_valid': sonnet_result['valid'],
                'sonnet_tokens': sonnet_result['estimated_tokens'],
                'sonnet_time': sonnet_result['processing_time'],
                'sonnet_error': sonnet_result['error'],
                
                # Haiku results
                'haiku_success': haiku_result['success'],
                'haiku_output': haiku_result['output'],
                'haiku_valid': haiku_result['valid'],
                'haiku_tokens': haiku_result['estimated_tokens'],
                'haiku_time': haiku_result['processing_time'],
                'haiku_error': haiku_result['error'],
                
                # Comparison metrics
                'both_successful': sonnet_result['success'] and haiku_result['success'],
                'both_valid': sonnet_result['valid'] and haiku_result['valid'],
                'outputs_identical': sonnet_result['output'] == haiku_result['output'] if sonnet_result['output'] and haiku_result['output'] else False,
                'sonnet_faster': sonnet_result['processing_time'] < haiku_result['processing_time']
            }
            
            self.results.append(result_entry)
            
            # Delay between expressions to be nice to API
            time.sleep(1)
        
        print("\nProcessing complete!")
    
    def generate_summary_stats(self):
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        total_expressions = len(self.results)
        
        # Success rates
        sonnet_successes = sum(1 for r in self.results if r['sonnet_success'])
        haiku_successes = sum(1 for r in self.results if r['haiku_success'])
        
        # Validity rates
        sonnet_valid = sum(1 for r in self.results if r['sonnet_valid'])
        haiku_valid = sum(1 for r in self.results if r['haiku_valid'])
        
        # Agreement
        both_successful = sum(1 for r in self.results if r['both_successful'])
        outputs_identical = sum(1 for r in self.results if r['outputs_identical'])
        
        # Performance
        avg_sonnet_time = sum(r['sonnet_time'] for r in self.results) / total_expressions
        avg_haiku_time = sum(r['haiku_time'] for r in self.results) / total_expressions
        
        # Costs
        sonnet_cost = (self.total_sonnet_tokens / 1000) * self.sonnet_cost_per_1k_tokens
        haiku_cost = (self.total_haiku_tokens / 1000) * self.haiku_cost_per_1k_tokens
        
        return {
            'total_expressions': total_expressions,
            'processing_date': datetime.now().isoformat(),
            
            # Success rates
            'sonnet_success_rate': sonnet_successes / total_expressions,
            'haiku_success_rate': haiku_successes / total_expressions,
            
            # Validity rates  
            'sonnet_validity_rate': sonnet_valid / total_expressions,
            'haiku_validity_rate': haiku_valid / total_expressions,
            
            # Agreement
            'both_successful_rate': both_successful / total_expressions,
            'output_agreement_rate': outputs_identical / both_successful if both_successful > 0 else 0,
            
            # Performance
            'avg_sonnet_processing_time': avg_sonnet_time,
            'avg_haiku_processing_time': avg_haiku_time,
            'haiku_speed_advantage': (avg_sonnet_time - avg_haiku_time) / avg_sonnet_time,
            
            # Cost analysis
            'total_sonnet_tokens': self.total_sonnet_tokens,
            'total_haiku_tokens': self.total_haiku_tokens,
            'estimated_sonnet_cost': sonnet_cost,
            'estimated_haiku_cost': haiku_cost,
            'cost_ratio_sonnet_to_haiku': sonnet_cost / haiku_cost if haiku_cost > 0 else float('inf'),
            'cost_per_expression_sonnet': sonnet_cost / total_expressions,
            'cost_per_expression_haiku': haiku_cost / total_expressions
        }
    
    def save_results(self, output_prefix="model_comparison"):
        """Save results in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate summary
        summary_stats = self.generate_summary_stats()
        
        # Save detailed JSON
        detailed_output = {
            'metadata': {
                'comparison_date': datetime.now().isoformat(),
                'total_expressions': len(self.results),
                'models_compared': ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022']
            },
            'summary_statistics': summary_stats,
            'detailed_results': self.results
        }
        
        json_file = f"{output_prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_output, f, indent=2, ensure_ascii=False)
        
        # Save CSV for easy analysis
        df = pd.DataFrame(self.results)
        csv_file = f"{output_prefix}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # Save summary report
        report_file = f"{output_prefix}_summary_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("CLAUDE MODEL COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Processing Date: {summary_stats['processing_date']}\n")
            f.write(f"Total Expressions: {summary_stats['total_expressions']}\n\n")
            
            f.write("SUCCESS RATES:\n")
            f.write(f"  Sonnet: {summary_stats['sonnet_success_rate']:.1%}\n")
            f.write(f"  Haiku:  {summary_stats['haiku_success_rate']:.1%}\n\n")
            
            f.write("VALIDITY RATES:\n")
            f.write(f"  Sonnet: {summary_stats['sonnet_validity_rate']:.1%}\n") 
            f.write(f"  Haiku:  {summary_stats['haiku_validity_rate']:.1%}\n\n")
            
            f.write("PERFORMANCE:\n")
            f.write(f"  Avg Sonnet Time: {summary_stats['avg_sonnet_processing_time']:.2f}s\n")
            f.write(f"  Avg Haiku Time:  {summary_stats['avg_haiku_processing_time']:.2f}s\n")
            f.write(f"  Haiku Speed Advantage: {summary_stats['haiku_speed_advantage']:.1%}\n\n")
            
            f.write("COST ANALYSIS:\n")
            f.write(f"  Sonnet Total Cost: ${summary_stats['estimated_sonnet_cost']:.4f}\n")
            f.write(f"  Haiku Total Cost:  ${summary_stats['estimated_haiku_cost']:.4f}\n")
            f.write(f"  Cost Ratio (S:H):  {summary_stats['cost_ratio_sonnet_to_haiku']:.1f}:1\n")
            f.write(f"  Cost/Expression (Sonnet): ${summary_stats['cost_per_expression_sonnet']:.4f}\n")
            f.write(f"  Cost/Expression (Haiku):  ${summary_stats['cost_per_expression_haiku']:.4f}\n\n")
            
            f.write("AGREEMENT:\n")
            f.write(f"  Both Successful: {summary_stats['both_successful_rate']:.1%}\n")
            f.write(f"  Output Agreement: {summary_stats['output_agreement_rate']:.1%}\n")
        
        print(f"\nResults saved:")
        print(f"  Detailed JSON: {json_file}")
        print(f"  CSV: {csv_file}")
        print(f"  Summary: {report_file}")
        
        return json_file, csv_file, report_file

def main():
    """Main execution function."""
    data_file = "data/cops_ref_test_sample_50.json"
    
    if not Path(data_file).exists():
        print(f"Error: {data_file} not found!")
        return
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Load data
    print("Loading COPS-Ref sample data...")
    expressions = comparator.load_cops_ref_data(data_file)
    print(f"Loaded {len(expressions)} expressions")
    
    # Show cost estimates
    estimated_cost = len(expressions) * (0.015 + 0.002)  # Rough estimate per expression
    print(f"Estimated total cost: ~${estimated_cost:.2f}")
    
    if input("\nProceed with comparison? (y/n): ").lower() != 'y':
        return
    
    # Run comparison
    start_time = time.time()
    comparator.compare_models(expressions)
    total_time = time.time() - start_time
    
    print(f"\nTotal processing time: {total_time:.1f}s")
    
    # Save results
    print("\nSaving results...")
    comparator.save_results()
    
    # Print quick summary
    summary = comparator.generate_summary_stats()
    print(f"\nQUICK SUMMARY:")
    print(f"  Sonnet Success: {summary['sonnet_success_rate']:.1%}")
    print(f"  Haiku Success:  {summary['haiku_success_rate']:.1%}")
    print(f"  Cost Difference: {summary['cost_ratio_sonnet_to_haiku']:.1f}x")
    print(f"  Agreement Rate: {summary['output_agreement_rate']:.1%}")

if __name__ == "__main__":
    main()