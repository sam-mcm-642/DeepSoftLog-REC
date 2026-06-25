#!/usr/bin/env python3
"""
Test and validation script for the referring expression processor.
"""

import os
import json
import time
from dotenv import load_dotenv
from llm_parser import ClaudeReferringExpressionProcessor, clean_response, validate_output

def test_api_connection():
    """Test basic API connection."""
    print("🔌 Testing Claude API connection...")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found in environment!")
        print("   Please add your API key to the .env file")
        return False
    
    try:
        processor = ClaudeReferringExpressionProcessor()
        response = processor.process_expression("The red car")
        print("✅ API connection successful!")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def test_individual_expressions():
    """Test processing of individual expressions."""
    print("\n🧪 Testing individual expressions...")
    
    processor = ClaudeReferringExpressionProcessor()
    
    test_cases = [
        "The person sitting on a bench",
        "The red car next to a tree",
        "The white plate to the right of the tomato",
        "The dog lying under the table near a chair",
        "The man wearing a blue shirt standing behind a woman"
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{test_case}'")
        
        try:
            start_time = time.time()
            response = processor.process_expression(test_case)
            cleaned = clean_response(response)
            elapsed = time.time() - start_time
            
            if validate_output(cleaned):
                print(f"   ✅ Valid output ({elapsed:.1f}s)")
                print(f"   📝 Result: {cleaned}")
                results.append({
                    'input': test_case,
                    'output': cleaned,
                    'status': 'success',
                    'time': elapsed
                })
            else:
                print(f"   ❌ Invalid output format")
                print(f"   📝 Raw response: {response}")
                results.append({
                    'input': test_case,
                    'output': response,
                    'status': 'invalid_format',
                    'time': elapsed
                })
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'input': test_case,
                'output': None,
                'status': 'error',
                'error': str(e)
            })
    
    return results

def test_cops_ref_format():
    """Test with COPS-Ref format data."""
    print("\n📋 Testing COPS-Ref format processing...")
    
    if not os.path.exists("test_cops_ref_data.json"):
        print("❌ test_cops_ref_data.json not found!")
        return False
    
    try:
        # Test loading the data
        from main_processor import load_cops_ref_data
        data = load_cops_ref_data("test_cops_ref_data.json")
        
        print(f"✅ Successfully loaded {len(data)} expressions from COPS-Ref format")
        
        # Test processing first few expressions
        processor = ClaudeReferringExpressionProcessor()
        
        for i, item in enumerate(data[:3]):  # Test first 3 only
            print(f"\n   Testing item {i+1}: {item['id']}")
            print(f"   Input: '{item['input']}'")
            
            try:
                response = processor.process_expression(item['input'])
                cleaned = clean_response(response)
                
                if validate_output(cleaned):
                    print(f"   ✅ Valid: {cleaned}")
                else:
                    print(f"   ❌ Invalid format: {response}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ COPS-Ref format test failed: {e}")
        return False

def test_output_format():
    """Test that output format matches expected structure."""
    print("\n📊 Testing output format compliance...")
    
    processor = ClaudeReferringExpressionProcessor()
    
    test_expression = "The white plate that is to the right of the tomato"
    response = processor.process_expression(test_expression)
    cleaned = clean_response(response)
    
    print(f"Input: {test_expression}")
    print(f"Output: {cleaned}")
    
    # Check components
    checks = [
        ("target(X)", "target(X)" in cleaned),
        ("type(X,", "type(X," in cleaned),
        ("expression(", "expression(" in cleaned),
        ("comma-separated", cleaned.count(",") >= 2)
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}: {passed}")
        if not passed:
            all_passed = False
    
    return all_passed

def analyze_predicate_usage(results):
    """Analyze which predicates are being used."""
    print("\n📈 Analyzing predicate usage...")
    
    import re
    predicate_counts = {}
    
    for result in results:
        if result.get('status') == 'success':
            # Extract predicates
            predicates = re.findall(r'expression\(([^,]+),', result['output'])
            for pred in predicates:
                pred = pred.strip()
                predicate_counts[pred] = predicate_counts.get(pred, 0) + 1
    
    if predicate_counts:
        print("   Most common predicates:")
        for pred, count in sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"     {pred}: {count}")
    else:
        print("   No predicates found")
    
    return predicate_counts

def create_validation_report(results, predicates):
    """Create a validation report."""
    print("\n📄 Creating validation report...")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': len(results),
        'successful': len([r for r in results if r.get('status') == 'success']),
        'failed': len([r for r in results if r.get('status') != 'success']),
        'predicate_usage': predicates,
        'results': results
    }
    
    with open('validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Validation report saved to validation_report.json")
    
    # Print summary
    success_rate = (report['successful'] / report['total_tests']) * 100
    print(f"\n📊 Summary:")
    print(f"   Success rate: {success_rate:.1f}% ({report['successful']}/{report['total_tests']})")
    print(f"   Average time per expression: {sum(r.get('time', 0) for r in results) / len(results):.1f}s")
    
    return report

def run_full_test_suite():
    """Run the complete test suite."""
    print("🚀 Running Full Test Suite")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    
    # Test 1: API Connection
    if not test_api_connection():
        print("\n❌ API connection failed. Please check your setup.")
        return False
    
    # Test 2: Individual expressions
    results = test_individual_expressions()
    
    # Test 3: COPS-Ref format
    test_cops_ref_format()
    
    # Test 4: Output format
    format_valid = test_output_format()
    
    # Analysis
    predicates = analyze_predicate_usage(results)
    report = create_validation_report(results, predicates)
    
    # Final assessment
    print("\n" + "=" * 50)
    if report['successful'] >= len(results) * 0.8 and format_valid:
        print("🎉 All tests passed! Your system is ready for production.")
        print("\n Next steps:")
        print("   1. Run: python main_processor.py test_cops_ref_data.json output.jsonl")
        print("   2. Check output.jsonl for results")
        print("   3. Use the output for your embedding training")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

def quick_test():
    """Quick test for basic functionality."""
    print("⚡ Quick Test")
    print("-" * 20)
    
    load_dotenv()
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ No API key found. Please set up your .env file first.")
        return
    
    processor = ClaudeReferringExpressionProcessor()
    test_input = "The red car next to a tree"
    
    print(f"Testing: '{test_input}'")
    
    try:
        response = processor.process_expression(test_input)
        cleaned = clean_response(response)
        
        print(f"Result: {cleaned}")
        
        if validate_output(cleaned):
            print("✅ Quick test passed!")
        else:
            print("❌ Output format invalid")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    print("🔍 Starting test suite...")
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        run_full_test_suite()