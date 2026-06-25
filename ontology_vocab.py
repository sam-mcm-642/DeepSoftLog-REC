#!/usr/bin/env python3
"""
Script to extract vocabulary from the ontology CSV file.
This helps track which terms need to be used in query generation.
"""

import pandas as pd
import json
from collections import Counter, defaultdict

def extract_ontology_vocabulary(ontology_path: str, output_path: str = None):
    """
    Extract all unique terms from the ontology and analyze their relationships.
    
    Args:
        ontology_path: Path to the ontology CSV file
        output_path: Optional path to save the vocabulary JSON file
    
    Returns:
        Dictionary containing vocabulary analysis
    """
    # Load ontology
    ontology_df = pd.read_csv(ontology_path)
    
    # Extract all unique terms
    all_terms = set()
    all_terms.update(ontology_df['subject'].unique())
    all_terms.update(ontology_df['object'].unique())
    
    # Count term frequencies (how connected each term is)
    term_connections = Counter()
    for _, row in ontology_df.iterrows():
        term_connections[row['subject']] += 1
        term_connections[row['object']] += 1
    
    # Group terms by relationship type
    relation_groups = defaultdict(set)
    for _, row in ontology_df.iterrows():
        relation_groups[row['relation']].add(row['subject'])
        relation_groups[row['relation']].add(row['object'])
    
    # Create vocabulary analysis
    vocab_analysis = {
        'total_terms': len(all_terms),
        'all_terms': sorted(list(all_terms)),
        'term_connections': dict(term_connections),
        'relation_groups': {k: sorted(list(v)) for k, v in relation_groups.items()},
        'most_connected_terms': dict(term_connections.most_common(20)),
        'least_connected_terms': dict(term_connections.most_common()[-20:]),
        'terms_by_relation_count': {
            relation: len(terms) for relation, terms in relation_groups.items()
        }
    }
    
    # Print summary
    print(f"Total unique terms in ontology: {len(all_terms)}")
    print(f"Relationship types: {list(relation_groups.keys())}")
    print(f"Most connected terms: {list(dict(term_connections.most_common(10)).keys())}")
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(vocab_analysis, f, indent=2)
        print(f"Vocabulary analysis saved to: {output_path}")
    
    return vocab_analysis

if __name__ == "__main__":
    ontology_path = "data/final_ontology.csv"
    output_path = "data/query/ontology_vocabulary.json"
    
    vocab_analysis = extract_ontology_vocabulary(ontology_path, output_path)