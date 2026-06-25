#!/usr/bin/env python3
"""
Prolog Query Structure Validator - Summary Mode

This script validates the structure of prolog-like queries from an LLM output JSON file
and outputs only summary statistics.

Usage: python validate_queries.py <json_file_path>
"""

import json
import re
import sys
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class Predicate:
    def __init__(self, name: str, args: List[str]):
        self.name = name
        self.args = args


def parse_predicates(query_str: str) -> List[str]:
    """Parse a query string into individual predicate strings."""
    predicates = []
    current = ''
    paren_count = 0
    
    for char in query_str:
        if char == '(':
            paren_count += 1
            current += char
        elif char == ')':
            paren_count -= 1
            current += char
        elif char == ',' and paren_count == 0:
            if current.strip():
                predicates.append(current.strip())
                current = ''
        else:
            current += char
    
    if current.strip():
        predicates.append(current.strip())
    
    return predicates


def parse_predicate(pred_str: str) -> Optional[Predicate]:
    """Parse a single predicate string into name and arguments."""
    match = re.match(r'^(\w+)\((.*)\)$', pred_str.strip())
    if not match:
        return None
    
    name, args_str = match.groups()
    
    args = []
    current = ''
    paren_count = 0
    
    for char in args_str:
        if char == '(':
            paren_count += 1
            current += char
        elif char == ')':
            paren_count -= 1
            current += char
        elif char == ',' and paren_count == 0:
            args.append(current.strip())
            current = ''
        else:
            current += char
    
    if current.strip():
        args.append(current.strip())
    
    return Predicate(name, args)


def validate_query(query_str: str, image_id: str) -> Tuple[bool, List[str]]:
    """
    Validate a single query according to the specified structure.
    
    Returns:
        (is_valid, list_of_issues)
    """
    predicate_strings = parse_predicates(query_str)
    predicates = [parse_predicate(ps) for ps in predicate_strings]
    predicates = [p for p in predicates if p is not None]
    
    issues = []
    
    # Check if we have at least 3 predicates
    if len(predicates) < 3:
        issues.append("Too few predicates")
        return False, issues
    
    # Count each type of predicate
    target_count = sum(1 for p in predicates if p.name == 'target')
    type_count = sum(1 for p in predicates if p.name == 'type')
    expression_count = sum(1 for p in predicates if p.name == 'expression')
    
    # Check exactly one target predicate
    if target_count != 1:
        if target_count == 0:
            issues.append("Missing target predicate")
        else:
            issues.append("Multiple target predicates")
    
    # Check exactly one type predicate
    if type_count != 1:
        if type_count == 0:
            issues.append("Missing type predicate")
        else:
            issues.append("Multiple type predicates")
    
    # Check at least one expression predicate
    if expression_count < 1:
        issues.append("Missing expression predicate")
    
    # Check first predicate is target
    if len(predicates) > 0 and predicates[0].name != 'target':
        issues.append("First predicate not target")
    
    # Check target predicate arguments
    if len(predicates) > 0 and predicates[0].name == 'target':
        target_pred = predicates[0]
        if len(target_pred.args) != 1:
            issues.append("Target predicate wrong argument count")
        elif target_pred.args[0] != 'X':
            issues.append("Target predicate argument not X")
    
    # Check second predicate is type
    if len(predicates) > 1 and predicates[1].name != 'type':
        issues.append("Second predicate not type")
    
    # Check type predicate arguments
    if len(predicates) > 1 and predicates[1].name == 'type':
        type_pred = predicates[1]
        if len(type_pred.args) != 2:
            issues.append("Type predicate wrong argument count")
        elif type_pred.args[0] != 'X':
            issues.append("Type predicate first argument not X")
    
    # Check expression predicates
    for i, pred in enumerate(predicates[2:], start=2):
        if pred.name != 'expression':
            issues.append("Non-expression predicate in expression position")
        elif len(pred.args) != 3:
            issues.append("Expression predicate wrong argument count")
    
    return len(issues) == 0, issues


def analyze_dataset(file_path: str):
    """Analyze the complete dataset and print summary results."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}': {e}")
        return
    
    if 'queries' not in data:
        print("Error: JSON file must contain a 'queries' key with list of query objects.")
        return
    
    queries = data['queries']
    
    # Validate all queries
    total_queries = len(queries)
    failed_queries = 0
    failure_reasons = defaultdict(int)
    failed_query_details = []
    
    for query_data in queries:
        query_str = query_data.get('query', '')
        image_id = query_data.get('image_id', 'unknown')
        
        is_valid, issues = validate_query(query_str, image_id)
        
        if not is_valid:
            failed_queries += 1
            failed_query_details.append((image_id, query_str, issues))
            for issue in issues:
                failure_reasons[issue] += 1
    
    # Print summary results
    print("=" * 50)
    print("QUERY VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total queries analyzed: {total_queries}")
    print(f"Valid queries: {total_queries - failed_queries}")
    print(f"Failed queries: {failed_queries}")
    print(f"Success rate: {((total_queries - failed_queries) / total_queries * 100):.1f}%")
    
    if failed_queries > 0:
        print(f"\nFAILURE BREAKDOWN:")
        print("-" * 30)
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"{reason}: {count}")
        
        print(f"\nPROBLEMATIC QUERIES:")
        print("-" * 50)
        for image_id, query_str, issues in failed_query_details:
            print(f"Image {image_id}:")
            print(f"  Query: {query_str}")
            print(f"  Issues: {', '.join(issues)}")
            print()
    else:
        print("\n✅ All queries passed validation!")


def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_queries.py <json_file_path>")
        print("\nExample: python validate_queries.py final_queries_edited.json")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyze_dataset(file_path)


if __name__ == "__main__":
    main()