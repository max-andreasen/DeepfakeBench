
"""
This file inputs the transformer model class and searches for
good hyper-parameters. Output the results and best performing parameters
in file -->
"""

import optuna

from train_transformer import Trainer


class ParameterSearch:
    def __init__(self, data_split_file, n_trials=50):
        self.data_split_file = data_split_file
        self.n_trials = n_trials
