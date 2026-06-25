"""
Specialized formatter for ontology relationship CSV files.
Ensures consistent camelCase formatting in ontology entries.
"""

import pandas as pd
import re
import os
from tqdm import tqdm


def format_string(s):
    """
    Format string to camelCase with:
    - First word lowercase
    - All subsequent words capitalized
    - No spaces or underscores
    
    Args:
        s (str): Input string to format
    
    Returns:
        str: Formatted string
    """
    if not s or not isinstance(s, str):
        return s
    
    # Special case for has_attribute and other common predicates
    special_cases = {
        "has_attribute": "hasAttribute",
        "next_to": "nextTo",
        "on_top_of": "onTopOf",
        "in_front_of": "inFrontOf",
        "to_the_right_of": "toTheRightOf",
        "to_the_left_of": "toTheLeftOf",
        "part_of": "partOf",
        "behind": "behind",  # No change needed, just for completeness
        "coffee_table": "coffeeTable",
        "telephone_set": "telephoneSet",
        "computer_monitor": "computerMonitor",
        "office_chair": "officeChair",
        "desk_chair": "deskChair",
        "dining_table": "diningTable",
        "coffee_mug": "coffeeMug",
        "drinking_glass": "drinkingGlass",
        "reading_lamp": "readingLamp",
        "writing_desk": "writingDesk"
    }
    
    # Check if this is a special case (case-insensitive)
    lower_s = s.lower()
    if lower_s in special_cases:
        return special_cases[lower_s]
    
    # Convert to lowercase first
    s = s.lower()
    
    # Replace underscores with spaces temporarily
    s = s.replace('_', ' ')
    
    # Split by spaces and other separators
    words = re.split(r'[\s-]+', s)
    
    # First word lowercase, capitalize first letter of subsequent words
    formatted = words[0]
    for word in words[1:]:
        if word:  # Skip empty words
            formatted += word[0].upper() + word[1:] if len(word) > 1 else word.upper()
    
    return formatted


def format_ontology_csv(input_path, output_path):
    """
    Process an ontology CSV to ensure consistent camelCase formatting.
    
    Args:
        input_path (str): Path to input ontology CSV file
        output_path (str): Path to save the formatted ontology CSV
    """
    print(f"Loading ontology from {input_path}...")
    
    try:
        # Load the ontology CSV
        ontology_df = pd.read_csv(input_path)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Get column names (they might vary between ontologies)
        columns = ontology_df.columns.tolist()
        
        # Standard column names for common ontology files
        subject_col = 'subject' if 'subject' in columns else columns[0]
        relation_col = 'relation' if 'relation' in columns else columns[1]
        object_col = 'object' if 'object' in columns else columns[2]
        
        print(f"Processing {len(ontology_df)} ontology entries...")
        print(f"Columns: {subject_col}, {relation_col}, {object_col}")
        
        # Format each column
        ontology_df[subject_col] = ontology_df[subject_col].apply(format_string)
        ontology_df[relation_col] = ontology_df[relation_col].apply(format_string)
        ontology_df[object_col] = ontology_df[object_col].apply(format_string)
        
        # Save the formatted ontology
        ontology_df.to_csv(output_path, index=False)
        print(f"Formatted ontology saved to {output_path}")
        
        # Verify formatting with a sample
        sample_size = min(10, len(ontology_df))
        sample = ontology_df.sample(sample_size)
        
        print(f"\nVerification - {sample_size} random samples:")
        for i, row in enumerate(sample.itertuples()):
            print(f"Entry {i+1}:")
            print(f"  {subject_col}: '{getattr(row, subject_col)}'")
            print(f"  {relation_col}: '{getattr(row, relation_col)}'")
            print(f"  {object_col}: '{getattr(row, object_col)}'")
            
            # Check for formatting issues
            for col in [subject_col, relation_col, object_col]:
                val = getattr(row, col)
                if isinstance(val, str) and ('_' in val or ' ' in val):
                    print(f"  WARNING: {col} '{val}' contains underscores or spaces!")
            print()
        
        return True
        
    except Exception as e:
        print(f"Error processing ontology file: {e}")
        return False


def analyze_ontology_patterns(input_path):
    """
    Analyze an ontology CSV to find common patterns that might need special handling.
    
    Args:
        input_path (str): Path to input ontology CSV file
    """
    print(f"Analyzing ontology patterns in {input_path}...")
    
    try:
        # Load the ontology CSV
        ontology_df = pd.read_csv(input_path)
        
        # Get column names
        columns = ontology_df.columns.tolist()
        
        # Standard column names for common ontology files
        subject_col = 'subject' if 'subject' in columns else columns[0]
        relation_col = 'relation' if 'relation' in columns else columns[1]
        object_col = 'object' if 'object' in columns else columns[2]
        
        # Look for patterns in each column
        for col in [subject_col, relation_col, object_col]:
            print(f"\nAnalyzing patterns in '{col}' column:")
            
            # Find terms with underscores
            underscore_terms = ontology_df[ontology_df[col].str.contains('_', na=False)][col].unique()
            if len(underscore_terms) > 0:
                print(f"Found {len(underscore_terms)} unique terms with underscores.")
                print("Sample terms with underscores:")
                for term in underscore_terms[:10]:  # Show at most 10 examples
                    formatted = format_string(term)
                    print(f"  '{term}' → '{formatted}'")
                
                if len(underscore_terms) > 10:
                    print(f"  ... and {len(underscore_terms) - 10} more.")
            else:
                print("No terms with underscores found.")
            
            # Find terms with spaces
            space_terms = ontology_df[ontology_df[col].str.contains(' ', na=False)][col].unique()
            if len(space_terms) > 0:
                print(f"Found {len(space_terms)} unique terms with spaces.")
                print("Sample terms with spaces:")
                for term in space_terms[:10]:  # Show at most 10 examples
                    formatted = format_string(term)
                    print(f"  '{term}' → '{formatted}'")
                
                if len(space_terms) > 10:
                    print(f"  ... and {len(space_terms) - 10} more.")
            else:
                print("No terms with spaces found.")
                
            # Find most common terms
            term_counts = ontology_df[col].value_counts().head(10)
            if not term_counts.empty:
                print("\nMost common terms:")
                for term, count in term_counts.items():
                    print(f"  '{term}': {count} occurrences")
        
        # Analyze relationship types
        if relation_col in ontology_df.columns:
            relation_types = ontology_df[relation_col].value_counts()
            print("\nRelationship types:")
            for rel, count in relation_types.items():
                print(f"  '{rel}': {count} occurrences")
        
        return True
        
    except Exception as e:
        print(f"Error analyzing ontology file: {e}")
        return False


def detect_formatting_issues(input_path):
    """
    Detect potential formatting issues in an ontology CSV file.
    
    Args:
        input_path (str): Path to input ontology CSV file
    """
    print(f"Checking for formatting issues in {input_path}...")
    
    try:
        # Load the ontology CSV
        ontology_df = pd.read_csv(input_path)
        
        # Get column names
        columns = ontology_df.columns.tolist()
        
        # Standard column names for common ontology files
        subject_col = 'subject' if 'subject' in columns else columns[0]
        relation_col = 'relation' if 'relation' in columns else columns[1]
        object_col = 'object' if 'object' in columns else columns[2]
        
        issues_found = False
        
        # Check each column for formatting issues
        for col in [subject_col, relation_col, object_col]:
            # Check for underscores
            underscore_mask = ontology_df[col].str.contains('_', na=False)
            if underscore_mask.any():
                issues_found = True
                underscore_count = underscore_mask.sum()
                print(f"WARNING: Found {underscore_count} entries with underscores in '{col}' column.")
                
                # Show some examples
                examples = ontology_df[underscore_mask][col].head(5).tolist()
                print(f"Examples: {examples}")
            
            # Check for spaces
            space_mask = ontology_df[col].str.contains(' ', na=False)
            if space_mask.any():
                issues_found = True
                space_count = space_mask.sum()
                print(f"WARNING: Found {space_count} entries with spaces in '{col}' column.")
                
                # Show some examples
                examples = ontology_df[space_mask][col].head(5).tolist()
                print(f"Examples: {examples}")
            
            # Check for inconsistent capitalization
            cap_issues = []
            sample = ontology_df[col].dropna().sample(min(1000, len(ontology_df))).tolist()
            
            for term in sample:
                if isinstance(term, str):
                    formatted = format_string(term)
                    if term != formatted and not ('_' in term or ' ' in term):
                        cap_issues.append((term, formatted))
            
            if cap_issues:
                issues_found = True
                print(f"WARNING: Found capitalization inconsistencies in '{col}' column.")
                print("Examples:")
                for original, formatted in cap_issues[:5]:
                    print(f"  '{original}' should be '{formatted}'")
        
        if not issues_found:
            print("No formatting issues detected.")
            
        return issues_found
        
    except Exception as e:
        print(f"Error checking formatting: {e}")
        return True  # Return True to indicate issues (error occurred)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ontology_formatter.py <input_ontology.csv> [output_ontology.csv]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # If output path is not provided, create one
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}_formatted.csv"
    
    # First, analyze the ontology to understand patterns
    print("=" * 80)
    print("STEP 1: Analyzing ontology patterns")
    print("=" * 80)
    analyze_ontology_patterns(input_path)
    
    # Check for formatting issues
    print("\n" + "=" * 80)
    print("STEP 2: Checking for formatting issues")
    print("=" * 80)
    detect_formatting_issues(input_path)
    
    # Format the ontology
    print("\n" + "=" * 80)
    print("STEP 3: Formatting the ontology")
    print("=" * 80)
    format_ontology_csv(input_path, output_path)
    
    print("\n" + "=" * 80)
    print(f"Processing complete! Formatted ontology saved to: {output_path}")
    print("=" * 80)