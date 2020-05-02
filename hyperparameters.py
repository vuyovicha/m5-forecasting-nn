from trainer import Trainer
import config
from model import ESRNN
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn


class BayesianOptimizationHP(nn.Module):
    def __init__(self, train_dataset, categories, data_loader):
        self.train_dataset = train_dataset
        self.categories = categories
        self.data_loader = data_loader

    def init_hyperparams(
        self, amount_of_epochs,
        learning_rate,
        optimization_step_size,
        gamma_coefficient,
        training_percentile,
        clip_value,
        LSTM_size,
        #dilations,
        input_window_length
    ):
        params = {
            'amount_of_epochs': int(amount_of_epochs),
            'learning_rate': learning_rate,
            'optimization_step_size': int(optimization_step_size),
            'gamma_coefficient': gamma_coefficient,
            'training_percentile': int(training_percentile),
            'clip_value': int(clip_value),
            'LSTM_size': int(LSTM_size),
            #'dilations': int(dilations),
            'input_window_length': int(input_window_length),
        }

        for name, value in params.items():
            print(name)
            print(value)
            print()

        model = ESRNN(len(self.train_dataset), self.categories, params)
        trainer = Trainer(model, self.data_loader, params)
        return -trainer.train_epochs()

    def bayesian_optimizer(self):
        optimizer = BayesianOptimization(
           f=self.init_hyperparams,
           pbounds=config.bounds,
           random_state=1)
        optimizer.maximize(init_points=10, n_iter=50)
