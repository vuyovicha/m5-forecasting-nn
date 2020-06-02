import numpy as np
import embedding_vectors_preparation
from config import params_init_val


def create_series_classification_dataset(series, series_prices, preprocessed_time_categories, snap_categories_numerical):
    prices_available_indexes = create_available_prices_indexes(series_prices)
    classification_dataset = []  # [previous_value, price, snap1, snap2, snap3, week_day, month, year, event1, event2, event3, event4, target]  #first goes data with 2 possible values and numerical, then - categorical
    for j in prices_available_indexes:
        target = 0 if series[j] == 0 else 1
        if j != 0:
            previous_value = 0 if series[j - 1] == 0 else 1
            classification_dataset.append([previous_value, series_prices[j].item()] + snap_categories_numerical[j] + preprocessed_time_categories[j] + [target])
    return np.array(classification_dataset)


# GLOBAL
def encode_labels(categories):
    encoded_categories = categories.copy()
    for category in range(len(encoded_categories[0])):
        category_unique_headers = embedding_vectors_preparation.create_category_unique_headers(encoded_categories, category)
        for unique_value_index in range(len(category_unique_headers)):
            for i in range(len(categories)):
                if encoded_categories[i][category] == category_unique_headers[unique_value_index]:
                    encoded_categories[i][category] = unique_value_index
    return encoded_categories


# GLOBAL
def preprocess_time_categories(time_categories):
    encoded_categories_slice = []
    for i in range(len(time_categories)):
        encoded_categories_slice.append(time_categories[i][:-3])
    encoded_categories = encode_labels(encoded_categories_slice)
    snap_categories_numerical = []
    preprocessed_time_categories = []
    for i in range(len(time_categories)):
        snap_categories = [int(j) for j in time_categories[i][-3:]]
        snap_categories_numerical.append(snap_categories)
        preprocessed_time_categories.append(encoded_categories[i])
    return preprocessed_time_categories, snap_categories_numerical


# GLOBAL
def create_val_targets(validation_days, dataset):
    val_targets = []
    for i in range(len(dataset)):
        current_val_targets = []
        for j in range(len(dataset[0])):
            if validation_days[0] <= j <= validation_days[-1]:
                if dataset[i, j] != 0:
                    current_val_target = 1
                else:
                    current_val_target = 0
                current_val_targets.append(current_val_target)
        val_targets.append(current_val_targets)
    return val_targets


def create_val_value(series_prices, preprocessed_time_categories, snap_categories_numerical, validation_day, rounded_output):
    preprocessed_time_category = preprocessed_time_categories[validation_day]
    snap_category_numerical = snap_categories_numerical[validation_day]
    j = validation_day
    val_value = [rounded_output, series_prices[j].item()] + snap_category_numerical + preprocessed_time_category
    return np.array(val_value)  # todo not sure if numpy is required here


def create_available_prices_indexes(series_prices):
    prices_available_indexes = []
    for j in range(len(series_prices)):
        if j < params_init_val['starting_validation_day']:
            if series_prices[j] != 0:
                prices_available_indexes.append(j)
    return prices_available_indexes

