import numpy as np
import statistics

def read_file(file_path):
    with open (file_path, "r") as file:
        data = file.read().split("\n")
    series = []
    categories = []
    for i in range(1, 31):
    #for i in range (1, len(data) - 1):  #why -1?
        line = data[i].split(',')  # deleted replace(...)
        categories.append([str(value) for value in line[1:6]])
        series.append(np.array([int(value) for value in line[6:] if value != ""]))
    return np.array(series), categories  # remove np from cat
    #return np.array(series), categories

def replace_zeroes(dataset):
    # print(*dataset[0], sep=", ")
    for i in range(len(dataset)):
        temp_list = [value for value in dataset[i] if value != 0]
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