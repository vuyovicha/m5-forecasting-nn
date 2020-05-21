import preprocessing
import create_prices_dataset
from model import ESRNN
from torch.utils.data import DataLoader
from dataset import DatasetTimeSeries
from trainer import Trainer
import config
from hyperparameters import BayesianOptimizationHP

calendar = "C:/Users/User/Desktop/m5 data/calendar.csv"
sell_prices = "C:/Users/User/Desktop/m5 data/sell_prices.csv"
sample_submission = "C:/Users/User/Desktop/m5 data/sample_submission.csv"
sales_train_validation = "C:/Users/User/Desktop/m5 data/sales_train_validation.csv"

train_dataset_read, categories = preprocessing.read_file_train(sales_train_validation)
preprocessing.replace_zeroes(train_dataset_read)
time_categories, weeks = preprocessing.read_and_preprocess_file_calendar(calendar)
sample_dataset = preprocessing.create_sample_dataset(sample_submission)
sell_prices_initial_data = preprocessing.read_sell_data(sell_prices)

validation_size = config.params_init_val['validation_size']
if not config.params_init_val['training_without_val_dataset']:
    val_dataset, train_dataset = preprocessing.create_val_dataset(train_dataset_read, validation_size)
else:
    train_dataset = train_dataset_read
    val_dataset = [[] for i in range(len(train_dataset))]

prices_dataset = create_prices_dataset.create_prices_dataset(len(train_dataset), weeks, sell_prices_initial_data)

#model = ESRNN(20, categories, time_categories, config.params_init_val, prices_dataset)  # this
model = ESRNN(len(train_dataset), categories, time_categories, config.params_init_val, prices_dataset)
entire_dataset = DatasetTimeSeries(train_dataset, val_dataset, categories, config.params_init_val['device'])
data_loader = DataLoader(entire_dataset, shuffle=False, batch_size=config.params_init_val['batch_size'])
Trainer(model, data_loader, config.params_init_val, train_dataset.shape[1], sample_dataset).train_epochs()  # this
#BayesianOptimizationHP(train_dataset, categories, data_loader).bayesian_optimizer()

