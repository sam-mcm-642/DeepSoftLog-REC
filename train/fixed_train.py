from deepsoftlog.experiments.countries.dataset import generate_prolog_files, get_test_dataloader, get_train_dataloader, get_val_dataloader
from deepsoftlog.training import load_program, load_config
from deepsoftlog.training.logger import WandbLogger
from deepsoftlog.training.loss import nll_loss, get_optimizer
from train.trainer import ReferringTrainer
from data.fixed_dataset import ReferringExpressionDataset
from data.fixed_dataloader import ReferringExpressionDataLoader


def get_train_val_split(scene_graph_file, queries_file=None, train_ratio=0.9, max_instances=None):
    """
    Create train/val split with improved dataset handling.
    Only initializes dataset once to avoid duplicate loading.
    """
    print(f"Creating train/val split with ratio {train_ratio}")
    
    # Create full dataset using the static method (initializes only once)
    full_dataset = ReferringExpressionDataLoader.create_dataset_from_files(
        scene_graph_file, queries_file, max_instances
    )
    
    total_instances = len(full_dataset.instances)
    print(f"Total dataset instances: {total_instances}")
    
    if total_instances == 0:
        raise ValueError("No valid instances were created from the data files!")
    
    # Calculate split point
    train_size = int(total_instances * train_ratio)
    
    # Create training dataset with first portion of instances
    train_instances = full_dataset.instances[:train_size]
    train_dataset = ReferringExpressionDataset(train_instances)
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Create validation dataset with remaining instances
    val_instances = full_dataset.instances[train_size:]
    val_dataset = ReferringExpressionDataset(val_instances)
    print(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def get_train_dataloader(cfg: dict):
    """Create training dataloader with improved error handling"""
    try:
        train_dataset, _ = get_train_val_split(
            cfg.get('data_file', "/Users/sammcmanagan/Desktop/Thesis/Model/data/vocab_filtered_vg_scene_graph_sample.csv"),
            cfg.get('queries_file', "/Users/sammcmanagan/Desktop/Thesis/Model/data/query/generated_queries.json"),
            train_ratio=cfg.get('train_ratio', 0.9),
            max_instances=cfg.get('max_instances', 500)
        )
        return ReferringExpressionDataLoader(train_dataset, batch_size=cfg['batch_size'])
    except Exception as e:
        print(f"ERROR creating train dataloader: {e}")
        raise


def get_val_dataloader(cfg: dict):
    """Create validation dataloader with improved error handling"""
    try:
        _, val_dataset = get_train_val_split(
            cfg.get('data_file', "/Users/sammcmanagan/Desktop/Thesis/Model/data/vocab_filtered_vg_scene_graph_sample.csv"),
            cfg.get('queries_file', "/Users/sammcmanagan/Desktop/Thesis/Model/data/query/generated_queries.json"),
            train_ratio=cfg.get('train_ratio', 0.9),
            max_instances=cfg.get('max_instances', 1500)
        )
        return ReferringExpressionDataLoader(val_dataset, batch_size=1)
    except Exception as e:
        print(f"ERROR creating val dataloader: {e}")
        raise


def get_eval_dataloader(cfg: dict):
    """Create evaluation dataloader for the evaluation pipeline"""
    try:
        eval_dataset = ReferringExpressionDataLoader.create_dataset_from_files(
            cfg.get('eval_data_file', cfg.get('data_file', 
                "/Users/sammcmanagan/Desktop/Thesis/Model/data/vocab_filtered_vg_scene_graph_sample.csv")),
            cfg.get('eval_queries_file', cfg.get('queries_file',
                "/Users/sammcmanagan/Desktop/Thesis/Model/data/query/generated_queries.json")),
            max_instances=cfg.get('max_eval_instances', None)
        )
        
        print(f"Evaluation dataset size: {len(eval_dataset)}")
        return ReferringExpressionDataLoader(eval_dataset, batch_size=1, shuffle=False)
    except Exception as e:
        print(f"ERROR creating eval dataloader: {e}")
        raise


def train(cfg):
    """Main training function with improved error handling"""
    try:
        cfg = load_config(cfg)
        eval_dataloader = get_val_dataloader(cfg)
        program = load_program(cfg, eval_dataloader)
        optimizer = get_optimizer(program.get_store(), cfg)
        
        logger = WandbLogger(cfg)
        trainer = ReferringTrainer(
            program, get_train_dataloader, nll_loss, optimizer,
            logger=logger,
            max_proofs=cfg['max_proofs'],
            max_branching=cfg['max_branching'],
            max_depth=cfg['max_depth'],
        )
        trainer.val_dataloader = eval_dataloader
        trainer.train(cfg)
        
    except Exception as e:
        print(f"ERROR in training: {e}")
        import traceback
        traceback.print_exc()
        raise


# Additional utility functions for debugging

def debug_dataset_creation(scene_graph_file, queries_file=None, max_instances=10):
    """
    Debug function to test dataset creation with limited instances.
    """
    print(f"=== DEBUG: Testing dataset creation ===")
    
    try:
        dataset = ReferringExpressionDataLoader.create_dataset_from_files(
            scene_graph_file, queries_file, max_instances
        )
        
        print(f"Dataset created successfully with {len(dataset)} instances")
        
        if len(dataset) > 0:
            print(f"Sample instance:")
            sample = dataset[0]
            print(f"  Query: {sample.query}")
            print(f"  Target: {sample.target}")
            print(f"  Scene graph objects: {len(sample.scene_graph.bounding_boxes)}")
            print(f"  Scene graph triplets: {len(sample.scene_graph.triplets)}")
        
        return dataset
        
    except Exception as e:
        print(f"ERROR in debug dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_data_files(scene_graph_file, queries_file):
    """
    Validate that data files exist and have expected format.
    """
    import os
    import json
    
    print(f"=== VALIDATING DATA FILES ===")
    
    # Check scene graph file
    if not os.path.exists(scene_graph_file):
        print(f"❌ Scene graph file not found: {scene_graph_file}")
        return False
    else:
        print(f"✅ Scene graph file found: {scene_graph_file}")
    
    # Check queries file
    if not os.path.exists(queries_file):
        print(f"❌ Queries file not found: {queries_file}")
        return False
    else:
        print(f"✅ Queries file found: {queries_file}")
        
        # Validate JSON structure
        try:
            with open(queries_file, 'r') as f:
                data = json.load(f)
                queries = data.get('queries', [])
                print(f"✅ Queries file contains {len(queries)} queries")
                
                if len(queries) > 0:
                    sample_query = queries[0]
                    required_fields = ['image_id', 'query', 'target']
                    missing_fields = [field for field in required_fields if field not in sample_query]
                    
                    if missing_fields:
                        print(f"❌ Sample query missing fields: {missing_fields}")
                        return False
                    else:
                        print(f"✅ Query structure looks correct")
                        
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in queries file: {e}")
            return False
    
    print(f"✅ All data files validated successfully")
    return True