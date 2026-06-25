#!/usr/bin/env python
"""
Script to analyze and fix formatting issues in ontology relationship CSV files.
Focuses specifically on addressing the underscore issue for embedding pretraining.
"""

import os
import sys
import argparse
from ontology_formatter import analyze_ontology_patterns, detect_formatting_issues, format_ontology_csv


def format_term(term: str) -> str:
    """
    Converts terms to the required format:
    - First word lowercase
    - Subsequent words capitalized
    - No spaces or symbols between words
    
    Example:
    "toilet_bowl" -> "toiletBowl"
    "random-access memory" -> "randomAccessMemory"
    """
    # Replace common separators with spaces to normalize
    normalized = term.replace('-', ' ').replace('_', ' ').replace('/', ' ')
    
    # Split into words
    words = normalized.split()
    
    # Format: first word lowercase, rest capitalized
    if not words:
        return ""
    
    formatted = words[0].lower()
    for word in words[1:]:
        formatted += word.capitalize()
    
    return formatted


def create_readable_df(filtered_df):
    readable_df = filtered_df.copy()
    
    # Function to get readable name and format it
    def get_readable_formatted_name(synset_name):
        try:
            if '.' in synset_name and len(synset_name.split('.')) == 3:
                synset = wn.synset(synset_name)
                name = synset.lemma_names()[0]
                return format_term(name)
            return format_term(synset_name)
        except:
            return format_term(synset_name)
    
    # Apply formatting to both subject and object columns
    readable_df['subject'] = readable_df['subject'].apply(get_readable_formatted_name)
    readable_df['object'] = readable_df['object'].apply(get_readable_formatted_name)
    
    return readable_df