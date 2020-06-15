import preprocessing
import create_prices_dataset
from model import ESRNN
from torch.utils.data import DataLoader
from dataset import DatasetTimeSeries
from trainer import Trainer
import config
import classification_preprocessing
from classification_trainer import GlobalClassificationTrainer
from hyperparameters import BayesianOptimizationHP
import numpy as np


# FILE'S PATHS
calendar = "C:/Users/User/Desktop/m5 data/calendar.csv"
sell_prices = "C:/Users/User/Desktop/m5 data/sell_prices.csv"
sample_submission = "C:/Users/User/Desktop/m5 data/sample_submission.csv"
sales_train_validation = "C:/Users/User/Desktop/m5 data/sales_train_validation.csv"

# READ DATASETS
sample_dataset = preprocessing.create_sample_dataset(sample_submission)
train_dataset_read, categories = preprocessing.read_file_train(sales_train_validation)
time_categories, weeks = preprocessing.read_and_preprocess_file_calendar(calendar)
sell_prices_initial_data = preprocessing.read_sell_data(sell_prices)

# CREATE PRICES DATASET AND SAVE IT
prices_dataset = create_prices_dataset.create_prices_dataset(len(train_dataset_read), weeks, sell_prices_initial_data)
# create_prices_dataset.save_prices_dataset(prices_dataset)
# prices_dataset = create_prices_dataset.read_saved_prices_dataset("C:/Users/User/Desktop/m5 data/PRICES_DATASET.csv")

# ENCODE CATEGORIES
preprocessed_time_categories, snap_categories_numerical = classification_preprocessing.preprocess_time_categories(time_categories)
encoded_categories = classification_preprocessing.encode_labels(categories)

# CREATE VALIDATION DATASET
validation_size = config.params_init_val['validation_size']
if not config.params_init_val['training_without_val_dataset']:
    val_dataset, train_dataset = preprocessing.create_val_dataset(train_dataset_read, validation_size)
else:
    train_dataset = train_dataset_read
    val_dataset = [[] for i in range(len(train_dataset))]

# CREATE DATASET AND DATA_LOADER
entire_dataset = DatasetTimeSeries(train_dataset, val_dataset, categories, config.params_init_val['device'])
data_loader = DataLoader(entire_dataset, shuffle=False, batch_size=config.params_init_val['batch_size'])

#INITIALIZE MODEL
model = ESRNN(len(train_dataset), encoded_categories, config.params_init_val, preprocessed_time_categories)

# SET THE NAME OF THE FILE WITH THE MODEL STATE IF NEEDED AND INITIALIZE TRAINER
needed_model_state = '' #'epoch_0_batch_0.tar'
Trainer(model, data_loader, config.params_init_val, train_dataset.shape[1], sample_dataset, needed_model_state).train_epochs()

# OPTIMIZE HYPERPARAMETERS
#BayesianOptimizationHP(len(train_dataset), categories, time_categories, used_days_dataset, predictions_indexes, predictions_lengths, zero_related_predictions_indexes, real_values_starting_indexes, data_loader, train_dataset.shape[1], sample_dataset).bayesian_optimizer()

