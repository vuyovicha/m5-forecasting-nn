
from bayes_opt import BayesianOptimization
from trainer import Trainer
import config
from model import ESRNN

#list_hyperparameters = []
#list_value_hyperparameters = []
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

    def bayesian_optimizer(self):
        bayesian_optimization.init_hyperparams(Trainer.amount_of_epochs,
        Trainer.learning_rate,
        Trainer.optimization_step_size,
        Trainer.gamma_coefficient,
        Trainer.ing_percentile,
        Trainer.clip_value,
        ESRNN.LSTM_size,
        #ESRNN.dilations,
        ESRNN.input_window_length,)
        model = ESRNN(len(self.train_dataset), self.categories, config.params)
        Trainer(model, self.data_loader, config.params).train_epochs()
        optimizer = BayesianOptimization(
           f=-Trainer.train_epochs(),
           pbounds=config.bounds,
            random_state=1,)
        optimizer.maximize(init_points=10, n_iter=50)

