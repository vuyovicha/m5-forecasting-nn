import numpy as np
import statistics


def read_file_train(file_path):
    with open(file_path, "r") as file:
        data = file.read().split("\n")
    series = []
    categories = []
    for i in range(1, 21):
    #for i in range (1, len(data) - 1):  # we exclude -1 because plain line is got when splitting by a \n
        line = data[i].split(',')  # deleted replace(...)
        categories.append([str(value) for value in line[1:6]])
        series.append(np.array([int(value) for value in line[6:] if value != ""]))
    return np.array(series), categories  # removed np from cat


def read_and_preprocess_file_calendar(file_path):
    with open(file_path, "r") as file:
        data = file.read().split("\n")
    time_categories = []

    for i in range(1, len(data) - 1):
        line = data[i].split(',')
        time_categories.append([str(value) for value in line])

    delete_indexes = [0, 1, 2, 6]
    for i in range(len(delete_indexes)):
        for j in range(len(time_categories)):
            del time_categories[j][delete_indexes[i] - i]

    for i in range(len(time_categories)):
        for j in range(len(time_categories[i])):
            if time_categories[i][j] == '':
                time_categories[i][j] = "NoValue"

    return time_categories

def replace_zeroes(dataset):
    # print(*dataset[0], sep=", ")
    for i in range(len(dataset)):
        temp_list = [value for value in dataset[i] if value != 0]  # todo do not count median using a val set
        current_median_value = statistics.median(temp_list)
        # if i == 0:
            # print(current_median_value)
        for j in range(dataset.shape[1]):
            if dataset[i, j] == 0:
                dataset[i, j] = current_median_value
        # if i == 0:
            # print()
            # print(*dataset[0], sep=", ")


def create_val_dataset(train_dataset_read, validation_size):
    val_dataset = []
    temp_train_dataset = []
    for i in range(len(train_dataset_read)):
        val_dataset.append(train_dataset_read[i][-validation_size:])
        temp_train_dataset.append(train_dataset_read[i][:-validation_size])
        # train_dataset[i] = train_dataset[i][:-validation_size]
    return np.array(val_dataset), np.array(temp_train_dataset)