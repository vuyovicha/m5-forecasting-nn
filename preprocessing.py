import numpy as np
import statistics
import config


def read_file_train(file_path):
    with open(file_path, "r") as file:
        data = file.read().split("\n")
    series = []
    categories = []
    for i in range(1, 2):
    #for i in range (1, len(data) - 1):  # we exclude -1 because plain line is got when splitting by a \n
        line = data[i].split(',')  # deleted replace(...)
        categories.append([str(value) for value in line[1:6]])
        series.append(np.array([float(value) for value in line[6:] if value != ""]))
    return np.array(series), categories  # removed np from cat


def read_and_preprocess_file_calendar(file_path):
    with open(file_path, "r") as file:
        data = file.read().split("\n")
    time_categories = []
    for i in range(1, len(data) - 1):
        line = data[i].split(',')
        time_categories.append([str(value) for value in line])
    weeks = []
    for i in range(len(time_categories)):
        weeks.append(time_categories[i][1])
    delete_indexes = [0, 1, 2, 6]
    for i in range(len(delete_indexes)):
        for j in range(len(time_categories)):
            del time_categories[j][delete_indexes[i] - i]
    for i in range(len(time_categories)):
        for j in range(len(time_categories[i])):
            if time_categories[i][j] == '':
                time_categories[i][j] = "NoValue"
    return time_categories, weeks


def replace_zeroes(dataset):
    for i in range(len(dataset)):
        if config.params_init_val['training_without_val_dataset']:
            temp_list = [value for value in dataset[i] if value != 0]
        else:
            temp_list = [value for value in dataset[i, :-config.params_init_val['validation_size']] if value != 0]  # todo do not count median using a val set
        current_median_value = statistics.median(temp_list)
        for j in range(dataset.shape[1] - config.params_init_val['validation_size']):
            if dataset[i, j] == 0:
                dataset[i, j] = current_median_value


def replace_zeroes_with_eps(dataset):
    for i in range(len(dataset)):
        for j in range(dataset.shape[1] - config.params_init_val['validation_size']):
            if dataset[i, j] == 0:
                dataset[i, j] = 1e-2


def create_val_dataset(train_dataset_read, validation_size):
    val_dataset = []
    temp_train_dataset = []
    for i in range(len(train_dataset_read)):
        val_dataset.append(train_dataset_read[i][-validation_size:])
        temp_train_dataset.append(train_dataset_read[i][:-validation_size])
    return np.array(val_dataset), np.array(temp_train_dataset)


def create_sample_dataset(sample_submission):
    file_path = sample_submission
    with open(file_path, "r") as file:
        data = file.read().split("\n")
    sample_dataset = []
    for i in range(len(data) - 1):
        line = data[i].split(',')
        sample_dataset.append([str(value) for value in line])
    return np.array(sample_dataset)


def read_sell_data(sell_prices):
    file_path = sell_prices
    with open(file_path, "r") as file:
        data = file.read().split("\n")
    sell_prices_initial_data = []
    for i in range(1, len(data) - 1):
        line = data[i].split(',')
        sell_prices_initial_data.append([str(value) for value in line])
    return sell_prices_initial_data


def read_zero_classifier_file(file):
    with open(file, "r") as file:
        data = file.read().split("\n")
    zero_classifier_predictions = []
    for i in range(0, len(data) - 1):
        line = data[i].split(',')
        zero_classifier_predictions.append(np.array([int(float(value)) for value in line]))  # nice trick here
    return np.array(zero_classifier_predictions)


def remove_zeros_and_pad_series(dataset):
    padded_dataset = []
    used_days_dataset = []
    real_values_starting_indexes = []
    for i in range(len(dataset)):
        current_series = []
        current_days = []
        for j in range(dataset.shape[1]):
            if dataset[i, j] != 0:
                current_series.append(dataset[i, j])
                current_days.append(j)
        padding_values = [0 for k in range(dataset.shape[1] - len(current_series))]
        real_values_starting_indexes.append(len(padding_values))
        padded_dataset.append(padding_values + current_series)
        used_days_dataset.append(current_days)
    return padded_dataset, np.array(used_days_dataset), real_values_starting_indexes


def get_non_zero_indexes_and_predictions_length(zero_classifier_predictions, starting_validation_day):
    predictions_indexes = []
    predictions_lengths = []
    zero_related_predictions_indexes = []
    for i in range(len(zero_classifier_predictions)):
        current_predictions_indexes = []
        current_zero_related_predictions_indexes = []
        for j in range(len(zero_classifier_predictions[0])):
            if zero_classifier_predictions[i, j] != 0:
                current_predictions_indexes.append(starting_validation_day + j)
                current_zero_related_predictions_indexes.append(j)
        predictions_indexes.append(current_predictions_indexes)
        zero_related_predictions_indexes.append(current_zero_related_predictions_indexes)
        predictions_lengths.append(len(current_predictions_indexes))
    return predictions_indexes, predictions_lengths, zero_related_predictions_indexes
