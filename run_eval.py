#!/usr/bin/env python3
"""
Convenience script to run evaluation with pretrained model
"""

import sys
from pathlib import Path
from eval import evaluate_model

def main():
    # Default paths - adjust these to your setup
    default_config = "eval/config.yaml"
    default_model = "results/thesis/store.pt"  # Adjust path to your saved model
    default_output = "evaluation_metrics_sg_new.csv"
    
    if len(sys.argv) == 1:
        # Use defaults
        config_path = default_config
        model_path = default_model
        output_path = default_output
        print(f"Using default paths:")
        print(f"  Config: {config_path}")
        print(f"  Model: {model_path}")
        print(f"  Output: {output_path}")
        
    elif len(sys.argv) == 3:
        config_path = sys.argv[1]
        model_path = sys.argv[2]
        output_path = default_output
        
    elif len(sys.argv) == 4:
        config_path = sys.argv[1]
        model_path = sys.argv[2]
        output_path = sys.argv[3]
        
    else:
        print("Usage:")
        print("  python run_eval.py                           # Use defaults")
        print("  python run_eval.py <config> <model>          # Specify config and model")
        print("  python run_eval.py <config> <model> <output> # Specify all paths")
        sys.exit(1)
    
    # Check if files exist
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
        
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Run evaluation
    try:
        evaluate_model(config_path, model_path, output_path)
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()