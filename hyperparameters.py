from trainer import Trainer
import config
from model import ESRNN
from bayes_opt import BayesianOptimization
import torch.nn as nn
import torch


class BayesianOptimizationHP(nn.Module):
    def __init__(self, train_dataset_len, categories, time_categories, used_days_dataset, predictions_indexes, predictions_lengths, zero_related_predictions_indexes, real_values_starting_indexes, data_loader, train_amount_of_days, sample_dataset):
        self.train_dataset_len = train_dataset_len
        self.categories = categories
        self.time_categories = time_categories
        self.used_days_dataset = used_days_dataset
        self.predictions_indexes = predictions_indexes
        self.predictions_lengths = predictions_lengths
        self.zero_related_predictions_indexes = zero_related_predictions_indexes
        self.real_values_starting_indexes = real_values_starting_indexes
        self.data_loader = data_loader
        self.train_amount_of_days = train_amount_of_days
        self.sample_dataset = sample_dataset

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
            'dilations': ((1, 7), (14, 28)),
            'LSTM_size': int(LSTM_size),
            'input_window_length': int(input_window_length),
            'batch_size': int(6),
            'validation_size': int(28),
            'device': ("cuda" if torch.cuda.is_available() else "cpu"),
            'output_window_length': int(28),
            'training_without_val_dataset': False,
            'starting_validation_day': int(1885),
            'classification_batch_size': int(50),
            'seasonality_parameter': int(7)
        }

        model = ESRNN(self.train_dataset_len, self.categories, self.time_categories, params, self.used_days_dataset, self.predictions_indexes, self.predictions_lengths, self.zero_related_predictions_indexes, self.real_values_starting_indexes)
        trainer = Trainer(model, self.data_loader, params, self.train_amount_of_days, self.sample_dataset, self.real_values_starting_indexes)
        return -trainer.train_epochs()

    def bayesian_optimizer(self):
        optimizer = BayesianOptimization(
           f=self.init_hyperparams,
           pbounds=config.bounds,
           random_state=1)
        optimizer.maximize(init_points=10, n_iter=50)
