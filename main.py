import pandas as pd
import numpy as np
import data_loading
from model import ESRNN
from torch.utils.data import DataLoader

calendar = pd.read_csv("C:/Users/User/Desktop/m5 data/calendar.csv")
sell_prices = pd.read_csv("C:/Users/User/Desktop/m5 data/sell_prices.csv")
sample_submission = pd.read_csv("C:/Users/User/Desktop/m5 data/sample_submission.csv")

train_dataset_read = data_loading.read_file("C:/Users/User/Desktop/m5 data/sales_train_validation.csv")

validation_size = 28  # check this!! validation dataset
val_dataset, train_dataset = data_loading.create_val_dataset(train_dataset_read, validation_size)

model = ESRNN(len(train_dataset))
data_loader = DataLoader()  # todo

