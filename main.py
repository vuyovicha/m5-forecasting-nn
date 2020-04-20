import pandas as pd
import preprocessing
from model import ESRNN
from torch.utils.data import DataLoader
from dataset import DatasetTimeSeries
from trainer import Trainer
import config
from hyperparameters import bayesian_optimization


calendar = pd.read_csv("C:/Users/User/Desktop/m5 data/calendar.csv")
sell_prices = pd.read_csv("C:/Users/User/Desktop/m5 data/sell_prices.csv")
sample_submission = pd.read_csv("C:/Users/User/Desktop/m5 data/sample_submission.csv")

train_dataset_read, categories = preprocessing.read_file("C:/Users/User/Desktop/m5 data/sales_train_validation.csv")
preprocessing.replace_zeroes(train_dataset_read)

validation_size = 28  # check this!! validation dataset
val_dataset, train_dataset = preprocessing.create_val_dataset(train_dataset_read, validation_size)

#model = ESRNN(30, categories, config.params_init_val)  # this
#model = ESRNN(len(train_dataset), categories, config.params_init_val)
entire_dataset = DatasetTimeSeries(train_dataset, val_dataset, categories)
#data_loader = DataLoader(entire_dataset, shuffle=True, batch_size=1024)
data_loader = DataLoader(entire_dataset, shuffle=True, batch_size=10)
#Trainer(model, data_loader, config.params_init_val).train_epochs()  # this
bayesian_optimization(train_dataset, categories, data_loader).bayesian_optimizer()

