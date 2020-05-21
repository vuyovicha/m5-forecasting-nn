import numpy as np
import embedding_vectors_preparation


def create_series_day_dataset(dataset, prices_dataset, encoded_categories, preprocessed_time_categories, snap_categories_numerical, starting_validation_day):
    zeroes_threshold = 15
    max_amounts_of_zeros = find_longest_zero_period(dataset, prices_dataset)
    last_day_zero_indexes = []
    is_more_zeros_than_threshold_list = []
    classification_dataset = []  # [series_index, day_index, days_since_last_zero, more_zeros_than_threshold, price, snap1, snap2, snap3, week_day, month, year, event1, event2, event3, event4, item, dept, cat, store, state]  #first goes data with 2 possible values and numerical, then - categorical
    for i in range(len(dataset)):
        last_zero_index = 0
        if max_amounts_of_zeros[i] > zeroes_threshold:
            is_more_zeros_than_threshold = 1
        else:
            is_more_zeros_than_threshold = 0
        is_more_zeros_than_threshold_list.append(is_more_zeros_than_threshold)
        for j in range(len(dataset[i])):
            if j < starting_validation_day:
                if prices_dataset[i, j] != 0:
                    if dataset[i, j] == 0:
                        target = 0
                    else:
                        target = 1
                    classification_dataset.append([i, j, j - last_zero_index, is_more_zeros_than_threshold, prices_dataset[i, j].item()] + snap_categories_numerical[j] + preprocessed_time_categories[j] + encoded_categories[i] + [target])
                if dataset[i, j] == 0:
                    last_zero_index = j
            elif j == starting_validation_day:
                last_day_zero_indexes.append(last_zero_index)
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
        encoded_categories_slice.append(time_categories[0][:-3])
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
    snap_category_numerical = snap_categories_numerical[validation_day]
    j = validation_day
    for i in range(len(dataset)):
        val_data.append([i, j, j - last_day_zero_indexes[i], is_more_zeros_than_threshold_list[i], prices_dataset[i, j].item()] + snap_category_numerical + preprocessed_time_category + encoded_categories[i])
    return np.array(val_data)


def compute_zero_prices_for_val(prices_dataset, predictions, current_index):
    for i in range(len(predictions)):
        if prices_dataset[i, current_index] == 0:
            predictions[i] = 0
    return predictions


def update_last_day_zero_indexes(last_day_zero_indexes, predictions, current_index):
    for i in range(len(predictions)):
        if predictions[i] == 0:
            last_day_zero_indexes[i] = current_index
    return last_day_zero_indexes




