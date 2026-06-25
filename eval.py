import sys
import csv
import json
import math
from pathlib import Path
from deepsoftlog.training import load_config, load_program
from deepsoftlog.parser.parser import parse_file
from deepsoftlog.embeddings.embedding_store import create_embedding_store
from data.dataset_sg import ReferringExpressionDataset
from data.dataloader import ReferringExpressionDataLoader
from train.evaluator_sg import ReferringEvaluator
from train.train import fix_pretrained_checkpoint
import os

def load_pretrained_model(model_path, program_path, config):
    """Load pretrained model with saved embeddings"""
    print(f"Loading pretrained model from: {model_path}")
    
    # Parse the base program with evaluation rule enabled
    program = parse_file(
        program_path,
        embedding_metric=config['embedding_metric'],
        semantics=config['semantics'],
    )
    
    # Load pretrained embeddings
    import torch
    checkpoint = torch.load(model_path, map_location=config['device'])
    
    # Create vocabulary from checkpoint
    from deepsoftlog.parser.vocabulary import Vocabulary
    vocab = Vocabulary()
    for const in checkpoint['vocabulary_constants']:
        vocab.add_constant(const)
    for functor_str in checkpoint['vocabulary_functors']:
        try:
            vocab.add_functor(eval(functor_str))
        except:
            pass
    
    # Create and load embedding store
    from deepsoftlog.embeddings.initialize_vector import Initializer
    from deepsoftlog.embeddings.nn_models import EmbeddingFunctor
    from deepsoftlog.embeddings.embedding_store import EmbeddingStore
    
    initializer = Initializer(EmbeddingFunctor, config['embedding_initialization'], config['embedding_dimensions'])
    store = EmbeddingStore(config['embedding_dimensions'], initializer, vocab)
    
    # Load state dict
    missing_keys, unexpected_keys = store.load_state_dict(checkpoint['embedding_store_state_dict'], strict=False)
    print(f"Loaded embeddings - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
    
    program.store = store
    return program


def get_eval_dataloader(config):
    """Create evaluation dataloader"""
    eval_dataset = ReferringExpressionDataset([])
    data_file = config.get('eval_data_file', config.get('data_file', 
        "/Users/sammcmanagan/Desktop/Thesis/Model/data/vg/sgg_scene_graphs_formatted.csv"))
    
    print(f"Loading evaluation data from: {data_file}")
    eval_dataset.generate_data_instances(data_file)
    
    # Limit instances if specified
    max_instances = config.get('max_eval_instances', len(eval_dataset.instances))

    if max_instances:
        if max_instances < len(eval_dataset.instances):
            eval_dataset.instances = eval_dataset.instances[:max_instances]
    
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    return ReferringExpressionDataLoader(eval_dataset, batch_size=1, shuffle=False)


def evaluate_model(config_path, model_path, output_path="evaluation_metrics_sgg.csv"):
    """Main evaluation function"""
    config = load_config(config_path)
    eval_dataloader = get_eval_dataloader(config)
    
    # Create initial program using config
    initial_program = load_program(config, eval_dataloader)
    print(f"🔍 evaluate_model: initial_program has {len(initial_program.store.constant_embeddings)} embeddings")
    
    # Create initial evaluator just to call load_pretrained_model
    temp_evaluator = ReferringEvaluator(
        program=initial_program,
        config=config,
        max_proofs=config.get('max_proofs', 10),
        max_branching=config.get('max_branching', 10), 
        max_depth=config.get('max_depth', 20),
    )
    print(f"🔍 evaluate_model: temp_evaluator.program has {len(temp_evaluator.program.store.constant_embeddings)} embeddings")
    
    # Fix checkpoint if needed
    fix_pretrained_checkpoint("pretrained_epoch_32_2025-08-11 17:00:30.pt", "pretrained_epoch_32_2025-08-11 17:00:30.pt")
    
    # Load pretrained program using the temp evaluator
    pretrained_program = temp_evaluator.load_pretrained_model(
        "pretrained_epoch_32_2025-08-11 17:00:30.pt", 
        initial_program_path=config.get('eval_program', config.get('initial_program_path', 'data/program/initial_program.pl'))
    )
    print(f"🔍 evaluate_model: pretrained_program has {len(pretrained_program.store.constant_embeddings)} embeddings")
    print(f"🔍 evaluate_model: pretrained_program id: {id(pretrained_program)}")
    
    # Create the actual evaluator with pretrained program
    print(f"🔍 evaluate_model: About to create final evaluator with pretrained_program...")
    evaluator = ReferringEvaluator(
        pretrained_program,
        config=config,
        max_proofs=config.get('max_proofs', 10),
        max_branching=config.get('max_branching', 10),
        max_depth=config.get('max_depth', 20),
    )
    print(f"🔍 evaluate_model: final evaluator.program has {len(evaluator.program.store.constant_embeddings)} embeddings")
    print(f"🔍 evaluate_model: evaluator.program id: {id(evaluator.program)}")
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluator.evaluate(eval_dataloader)
    
    # Save results to CSV
    evaluator.save_results_to_csv(results, output_path)
    
    # Print summary statistics
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_path}")
    print(f"Total instances evaluated: {len(results)}")
    
    # Calculate accuracy if ground truth available
    if any('ground_truth_object' in r for r in results):
        correct = sum(1 for r in results if r.get('predicted_object') == r.get('ground_truth_object'))
        accuracy = correct / len(results)
        print(f"Accuracy: {accuracy:.3f} ({correct}/{len(results)})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval.py <config_path> <model_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    model_path = sys.argv[2]
    # model_path = "pretrained_epoch_30_2025-08-11 15:42:40.pt"
    evaluate_model(config_path, model_path)