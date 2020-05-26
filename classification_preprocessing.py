import numpy as np
import embedding_vectors_preparation
import random
from config import params_init_val


"""def create_series_day_dataset(dataset, prices_dataset, encoded_categories, preprocessed_time_categories, snap_categories_numerical, starting_validation_day):
    zeroes_threshold = 15
    max_amounts_of_zeros = find_longest_zero_period(dataset, prices_dataset)
    last_zero_indexes = create_zero_days_indexes(dataset, starting_validation_day)
    is_more_zeros_than_threshold_list = []
    prices_available_indexes = create_available_prices_indexes(prices_dataset)
    random_sample_indexes, random_sample_last_zero_indexes = create_random_sample_indexes(prices_available_indexes, last_zero_indexes)
    classification_dataset = []  # [series_index, day_index, days_since_last_zero, more_zeros_than_threshold, price, snap1, snap2, snap3, week_day, month, year, event1, event2, event3, event4, item, dept, cat, store, state]  #first goes data with 2 possible values and numerical, then - categorical
    for i in range(len(dataset)):
        is_more_zeros_than_threshold = 1 if max_amounts_of_zeros[i] > zeroes_threshold else 0
        is_more_zeros_than_threshold_list.append(is_more_zeros_than_threshold)
        current_random_sample_last_zero_index = 0
        for random_index in random_sample_indexes[i]:
            last_zero_index = random_sample_last_zero_indexes[i][current_random_sample_last_zero_index]
            j = prices_available_indexes[i][random_index]
            target = 0 if dataset[i, j] == 0 else 1
            classification_dataset.append([i, j, j - last_zero_index, is_more_zeros_than_threshold, prices_dataset[i, j].item()] + snap_categories_numerical[j] + preprocessed_time_categories[j] + encoded_categories[i] + [target])
            current_random_sample_last_zero_index += 1
    last_day_zero_indexes = get_last_train_dataset_zero_index(last_zero_indexes)
    return np.array(classification_dataset), last_day_zero_indexes, is_more_zeros_than_threshold_list"""


def create_series_day_dataset(dataset, prices_dataset, encoded_categories, preprocessed_time_categories, snap_categories_numerical, starting_validation_day):
    last_zero_indexes = create_zero_days_indexes(dataset, starting_validation_day)
    is_more_zeros_than_threshold_list = []
    prices_available_indexes = create_available_prices_indexes(prices_dataset)
    random_sample_indexes, random_sample_last_zero_indexes = create_random_sample_indexes(prices_available_indexes, last_zero_indexes)
    classification_dataset = []  # [series_index, day_index, days_since_last_zero, more_zeros_than_threshold, price, snap1, snap2, snap3, week_day, month, year, event1, event2, event3, event4, item, dept, cat, store, state]  #first goes data with 2 possible values and numerical, then - categorical
    for i in range(len(dataset)):
        is_more_zeros_than_threshold_list.append(0)
        current_random_sample_last_zero_index = 0
        for random_index in random_sample_indexes[i]:
            last_zero_index = random_sample_last_zero_indexes[i][current_random_sample_last_zero_index]
            j = prices_available_indexes[i][random_index]
            target = 0 if dataset[i, j] == 0 else 1
            if j != 0:
                previous_value = 0 if dataset[i, j - 1] == 0 else 1
                classification_dataset.append([i, j, j - last_zero_index, previous_value, prices_dataset[i, j].item()] + snap_categories_numerical[j] + preprocessed_time_categories[j] + encoded_categories[i] + [target])
            current_random_sample_last_zero_index += 1
    last_day_zero_indexes = get_last_train_dataset_zero_index(last_zero_indexes)
    return np.array(classification_dataset), last_day_zero_indexes, is_more_zeros_than_threshold_list

def find_longest_zero_period(dataset, prices_dataset):
    max_amounts_of_zeros = []
    for i in range(len(dataset)):
        current_amount_of_zeros = 0
        current_max_amount_of_zeros = 0
        for j in range(len(dataset[i])):
            if prices_dataset[i, j] != 0 and dataset[i, j] == 0:
                current_amount_of_zeros += 1
            else:
                if current_amount_of_zeros > current_max_amount_of_zeros:
                    current_max_amount_of_zeros = current_amount_of_zeros
                current_amount_of_zeros = 0
        max_amounts_of_zeros.append(current_max_amount_of_zeros)
    return max_amounts_of_zeros


def encode_labels(categories):
    encoded_categories = categories.copy()
    for category in range(len(encoded_categories[0])):
        category_unique_headers = embedding_vectors_preparation.create_category_unique_headers(encoded_categories, category)
        for unique_value_index in range(len(category_unique_headers)):
            for i in range(len(categories)):
                if encoded_categories[i][category] == category_unique_headers[unique_value_index]:
                    encoded_categories[i][category] = unique_value_index
    return encoded_categories


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


# NOT USING THIS FOR NOW
def create_classification_train_val_data(classification_dataset, validation_days):
    classification_train_data = []
    classification_val_data = []
    for i in range(len(classification_dataset)):
        if classification_dataset[i, 1] in validation_days:
            classification_val_data.append(classification_dataset[i])
        else:
            classification_train_data.append(classification_dataset[i])
    return np.array(classification_train_data), np.array(classification_val_data)


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
    return np.array(val_targets)


def create_val_data(dataset, prices_dataset, encoded_categories, preprocessed_time_categories, snap_categories_numerical, validation_day, last_day_zero_indexes, is_more_zeros_than_threshold_list):
    val_data = []
    preprocessed_time_category = preprocessed_time_categories[validation_day]
    #print(preprocessed_time_category)
    #print(validation_day)
    snap_category_numerical = snap_categories_numerical[validation_day]
    j = validation_day
    for i in range(len(dataset)):
        val_data.append([i, j, j - last_day_zero_indexes[i], is_more_zeros_than_threshold_list[i], prices_dataset[i, j].item()] + snap_category_numerical + preprocessed_time_category + encoded_categories[i])
    return np.array(val_data)


def compute_zero_prices_for_val(prices_dataset, predictions, current_index):
    new_predictions = predictions.clone()
    for i in range(len(predictions)):
        if prices_dataset[i, current_index] == 0:
            new_predictions[i] = 0
    return new_predictions


def update_last_day_zero_indexes(last_day_zero_indexes, predictions, current_index):
    for i in range(len(predictions)):
        if predictions[i] == 0:
            last_day_zero_indexes[i] = current_index
    return last_day_zero_indexes


def create_available_prices_indexes(prices_dataset):
    prices_available_indexes = []
    for i in range(len(prices_dataset)):
        current_price_indexes = []
        for j in range(prices_dataset.shape[1]):
            if j < params_init_val['starting_validation_day']:
                if prices_dataset[i, j] != 0:
                    current_price_indexes.append(j)
        prices_available_indexes.append(current_price_indexes)
    return prices_available_indexes


def create_random_sample_indexes(prices_available_indexes, last_zero_indexes):
    random_sample_indexes = []
    random_sample_last_zero_indexes = []
    for i in range(len(prices_available_indexes)):
        if len(prices_available_indexes[i]) > params_init_val['amount_of_values_per_series']:
            random_sample_indexes.append(random.sample(range(0, len(prices_available_indexes[i])), params_init_val['amount_of_values_per_series']))
        else:
            random_sample_indexes.append(random.sample(range(0, len(prices_available_indexes[i])), len(prices_available_indexes[i])))
        current_random_sample_last_zero_indexes = []
        for j in random_sample_indexes[-1]:
            current_random_sample_last_zero_indexes.append(last_zero_indexes[i][prices_available_indexes[i][j]])
        random_sample_last_zero_indexes.append(current_random_sample_last_zero_indexes)
    return random_sample_indexes, random_sample_last_zero_indexes


def create_zero_days_indexes(dataset, starting_validation_day):
    last_zero_indexes = []
    for i in range(len(dataset)):
        last_zero_index = 0
        current_last_zero_indexes = []
        for j in range(len(dataset[i])):
            if j <= starting_validation_day:
                if dataset[i, j] == 0:
                    last_zero_index = j
                current_last_zero_indexes.append(last_zero_index)
        last_zero_indexes.append(current_last_zero_indexes)
    return last_zero_indexes


def get_last_train_dataset_zero_index(last_zero_indexes):
    last_day_zero_indexes = []
    for i in range(len(last_zero_indexes)):
        last_day_zero_indexes.append(last_zero_indexes[i][-1])
    return last_day_zero_indexes
