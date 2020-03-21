import numpy as np


def read_file(file_path):
    series = []
    with open (file_path, "r") as file:
        data = file.read().split("\n")

    for i in range (1, len(data) - 1):  #why -1?
        line = data[i].split(',')  # deleted replace(...)
        series.append(np.array([int(value) for value in line[6:] if value != ""]))

    return np.array(series)


def create_val_dataset(train_dataset_read, validation_size):
    val_dataset = []
    temp_train_dataset = []
    for i in range(len(train_dataset_read)):
        val_dataset.append(train_dataset_read[i][-validation_size:])
        temp_train_dataset.append(train_dataset_read[i][:-validation_size])
        # train_dataset[i] = train_dataset[i][:-validation_size]

    return np.array(val_dataset), np.array(temp_train_dataset)