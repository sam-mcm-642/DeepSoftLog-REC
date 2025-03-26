from deepsoftlog.experiments.countries.dataset import generate_prolog_files, get_test_dataloader, get_train_dataloader, get_val_dataloader
from deepsoftlog.training import load_program, load_config
from deepsoftlog.training.logger import WandbLogger
from deepsoftlog.training.loss import nll_loss, get_optimizer
from train.trainer import ReferringTrainer
from data.dataset import ReferringExpressionDataset
from data.dataloader import ReferringExpressionDataLoader


dataset = ReferringExpressionDataset([])
dataset.generate_data_instances("/Users/sammcmanagan/Desktop/Thesis/Model/data/sg/sample_scenes.csv")
print(F"DATASET TYPE: {type(dataset)}")
print(F"DATASET TYPE: {type(dataset[0])}")
train_dataset = ReferringExpressionDataset(dataset[:3])
val_dataset = ReferringExpressionDataset(dataset[4:5])


print(f"type(train_dataset): {type(train_dataset)}")


def get_train_dataloader(cfg: dict):
    return ReferringExpressionDataLoader(train_dataset)

def get_val_dataloader():
    print(f"VAL DATASET SIZE: {len(val_dataset)}", val_dataset)
    return ReferringExpressionDataLoader(val_dataset)


def train(cfg):
    cfg = load_config(cfg)
    eval_dataloader = get_val_dataloader()
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
