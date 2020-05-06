import pandas as pd
import preprocessing
from model import ESRNN
from torch.utils.data import DataLoader
from dataset import DatasetTimeSeries
from trainer import Trainer
import config
from hyperparameters import BayesianOptimizationHP

"""""
from pinball_loss import PinballLoss
import torch
crtiterion = PinballLoss(0.45, 1, 'cpu')
print(crtiterion(torch.rand(5), torch.rand(5)))
"""""

calendar = "C:/Users/User/Desktop/m5 data/calendar.csv"
sell_prices = "C:/Users/User/Desktop/m5 data/sell_prices.csv"
sample_submission = "C:/Users/User/Desktop/m5 data/sample_submission.csv"

train_dataset_read, categories = preprocessing.read_file_train("C:/Users/User/Desktop/m5 data/sales_train_validation.csv")
preprocessing.replace_zeroes(train_dataset_read)
time_categories = preprocessing.read_and_preprocess_file_calendar(calendar)

validation_size = 28  # check this!! validation dataset
val_dataset, train_dataset = preprocessing.create_val_dataset(train_dataset_read, validation_size)

model = ESRNN(20, categories, time_categories, config.params_init_val)  # this
#model = ESRNN(len(train_dataset), categories, config.params_init_val)
entire_dataset = DatasetTimeSeries(train_dataset, val_dataset, categories)
# change batch size in config
data_loader = DataLoader(entire_dataset, shuffle=False, batch_size=config.params_init_val['batch_size'])  # shuffle = false because we need to generate ordered
Trainer(model, data_loader, config.params_init_val).train_epochs()  # this
#BayesianOptimizationHP(train_dataset, categories, data_loader).bayesian_optimizer()

