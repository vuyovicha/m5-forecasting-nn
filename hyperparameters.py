
from bayes_opt import BayesianOptimization
from trainer import Trainer
import config
from model import ESRNN

class bayesian_optimization():
    def __init__(self, train_dataset, categories, data_loader):
        self.train_dataset = train_dataset
        self.categories = categories
        self.data_loader = data_loader
    def init_hyperparams(amount_of_epochs,
        learning_rate,
        optimization_step_size,
        gamma_coefficient,
        training_percentile,
        clip_value,
        LSTM_size,
        #dilations,
        input_window_length,):
        config.params = {
            'amount_of_epochs':int(amount_of_epochs),
            'learning_rate':learning_rate,
            'optimization_step_size':int(optimization_step_size),
            'gamma_coefficient':gamma_coefficient,
            'training_percentile':int(training_percentile),
            'clip_value':int(clip_value),
            'LSTM_size':int(LSTM_size),
            #'dilations' :int(dilations),
            'input_window_length' :int(input_window_length),
        }
        model = ESRNN(len(self.train_dataset), self.categories, config.params)
        trainer = Trainer(model, self.data_loader, config.params)
        return -trainer.train_epochs()

    def bayesian_optimizer(self):
        optimizer = BayesianOptimization(
           f=self.init_hyperparams,
           pbounds=config.bounds,
           random_state=1,)
        optimizer.maximize(init_points=10, n_iter=50)

