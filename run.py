import sys
from train.train import train
from deepsoftlog.data import dataset
from deepsoftlog.logic.soft_term import SoftTerm, TensorTerm

def main(experiment_name, config_file):
    train_functions = {'referring_expression': train}
    assert experiment_name in train_functions.keys(), f"Experiment name must be one of {tuple(train_functions.keys())}"
    return train_functions[experiment_name](config_file)


if __name__ == "__main__":
    main(*sys.argv[1:])