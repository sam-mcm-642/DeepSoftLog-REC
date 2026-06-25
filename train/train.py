from deepsoftlog.experiments.countries.dataset import generate_prolog_files, get_test_dataloader, get_train_dataloader, get_val_dataloader
from deepsoftlog.training import load_program, load_config
from deepsoftlog.training.logger import WandbLogger
from deepsoftlog.training.loss import nll_loss, get_optimizer
from train.trainer import ReferringTrainer
from data.dataset import ReferringExpressionDataset
from data.dataloader import ReferringExpressionDataLoader


def get_train_val_split(file_path, train_ratio=0.9, max_instances=None):
    # Create an empty dataset to populate
    full_dataset = ReferringExpressionDataset([])

    # Generate data instances from the file
    print(f"Loading data from: {file_path}")
    full_dataset.generate_data_instances(file_path)

    # Report the total number of instances
    total_instances = len(full_dataset.instances)
    print(f"Total dataset instances: {total_instances}")

    if total_instances == 0:
        raise ValueError("No instances were generated from the data file!")

    # Limit to max_instances if specified
    if max_instances and max_instances < total_instances:
        print(f"Limiting dataset to {max_instances} instances (out of {total_instances})")
        # Take a slice of the instances
        limited_instances = full_dataset.instances[:max_instances]
        total_instances = max_instances
    else:
        limited_instances = full_dataset.instances

    # Calculate split point
    train_size = int(total_instances * train_ratio)

    # Create training dataset with first portion of instances
    train_instances = limited_instances[:train_size]
    train_dataset = ReferringExpressionDataset(train_instances)
    print(f"Training dataset size: {len(train_dataset)}")

    # Create validation dataset with remaining instances
    val_instances = limited_instances[train_size:]
    val_dataset = ReferringExpressionDataset(val_instances)
    print(f"Validation dataset size: {len(val_dataset)}")

    return train_dataset, val_dataset

def get_train_dataloader(cfg: dict):
    train_dataset, _ = get_train_val_split(
        cfg.get('data_file', "data/vocab_filtered_vg_scene_graph_sample.csv"),
        train_ratio=cfg.get('train_ratio', 0.9),
        max_instances=cfg.get('max_instances', 500)
    )
    return ReferringExpressionDataLoader(train_dataset, batch_size=cfg['batch_size'])

def get_val_dataloader(cfg: dict):
    _, val_dataset = get_train_val_split(
        cfg.get('data_file', "data/vocab_filtered_vg_scene_graph_sample.csv"),
        train_ratio=cfg.get('train_ratio', 0.9),
        max_instances=cfg.get('max_instances', 1500)
    )
    return ReferringExpressionDataLoader(val_dataset, batch_size=1)

def train(cfg):
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
