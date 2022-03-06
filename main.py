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
import torch
import random

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
#create_prices_dataset.save_prices_dataset(prices_dataset)
#prices_dataset = create_prices_dataset.read_saved_prices_dataset("C:/Users/User/Desktop/m5 data/PRICES_DATASET.csv")


# ENCODE CATEGORIES FOR CLASSIFICATION
preprocessed_time_categories, snap_categories_numerical = classification_preprocessing.preprocess_time_categories(time_categories)

# CREATE LIST OF VALIDATION DAYS
starting_validation_day = config.params_init_val['starting_validation_day']
validation_days = [starting_validation_day + i for i in range(config.params_init_val['output_window_length'])]

# CREATE VALIDATION TARGETS
if not config.params_init_val['training_without_val_dataset']:
    val_targets = classification_preprocessing.create_val_targets(validation_days, train_dataset_read)
else:
    val_targets = [[] for i in range(len(train_dataset_read))]

# INITIALIZE GLOBAL CLASSIFICATION TRAINER
"""GlobalClassificationTrainer(train_dataset_read, prices_dataset, preprocessed_time_categories, snap_categories_numerical, val_targets, config.params_init_val).train_series_classifiers()
"""

""""" REGRESSION FROM THIS POINT """""

""""
random_binary_list = []
for i in range(len(train_dataset_read)):
    random_number = random.randint(2, 10)
    random_binary_list.append(torch.from_numpy(np.random.choice([0.0, 1.0], size=28, p=[1./random_number, (random_number - 1)/random_number])))
random_binary_list_cat = torch.cat([i.unsqueeze(0) for i in random_binary_list])
file_name = "CLASSIFICATION.csv"
np.savetxt(file_name, val_targets, delimiter=',', fmt="%s")"""

# PREPROCESS CLASSIFIER ZERO PREDICTIONS FILE
zero_classifier_predictions = preprocessing.read_zero_classifier_file("C:/Users/User/Desktop/project 2.0/CLASSIFICATION.csv")
predictions_indexes, predictions_lengths, zero_related_predictions_indexes = preprocessing.get_non_zero_indexes_and_predictions_length(zero_classifier_predictions, starting_validation_day)

# CREATE VALIDATION DATASET
validation_size = config.params_init_val['validation_size']
if not config.params_init_val['training_without_val_dataset']:
    val_dataset, train_dataset = preprocessing.create_val_dataset(train_dataset_read, validation_size)
else:
    train_dataset = train_dataset_read
    val_dataset = [[] for i in range(len(train_dataset))]

# PAD SERIES - REMOVE ALL ZEROS
padded_dataset, used_days_dataset, real_values_starting_indexes = preprocessing.remove_zeros_and_pad_series(train_dataset)

# CREATE DATASET AND DATA_LOADER
entire_dataset = DatasetTimeSeries(train_dataset, val_dataset, categories, config.params_init_val['device'])
data_loader = DataLoader(entire_dataset, shuffle=False, batch_size=config.params_init_val['batch_size'])

# INITIALIZE MODEL
model = ESRNN(5, categories, time_categories, config.params_init_val, used_days_dataset, predictions_indexes, predictions_lengths, zero_related_predictions_indexes, real_values_starting_indexes)  # len(train_dataset) instead of 20

# INITIALIZE TRAINER
Trainer(model, data_loader, config.params_init_val, train_dataset.shape[1], sample_dataset, real_values_starting_indexes).train_epochs()

# OPTIMIZE HYPERPARAMETERS
# BayesianOptimizationHP(train_dataset, categories, data_loader).bayesian_optimizer()

