import torch
import numpy as np


def find_index(given_week, weeks):
    for i in range(len(weeks)):
        if int(weeks[i]) == given_week:
            return i
    return False


def create_prices_dataset(train_dataset_len, weeks, sell_prices_initial_data):
    prices_dataset = torch.zeros([train_dataset_len, len(weeks)])
    #prices_dataset = torch.zeros([6000, len(weeks)])
    current_item = 0
    #for i in range(len(sell_prices_initial_data)):
    for i in range(6000):
        current_week_starting_index = find_index(int(sell_prices_initial_data[i][2]), weeks)
        for j in range(7):
            if current_week_starting_index + j < len(weeks):
                prices_dataset[current_item, current_week_starting_index + j] = float(sell_prices_initial_data[i][3])
        if i + 1 != len(sell_prices_initial_data):
            if sell_prices_initial_data[i][1] != sell_prices_initial_data[i + 1][1]:
                current_item += 1
        if current_item == train_dataset_len:  # todo remove this
            break
    return prices_dataset


def save_prices_dataset(prices_dataset):
    file_name = "PRICES_DATASET.csv"
    np.savetxt(file_name, prices_dataset.cpu(), delimiter=',', fmt="%s")


def read_saved_prices_dataset(file_path):
    with open(file_path, "r") as file:
        data = file.read().split("\n")
    prices_dataset_list = []
    for i in range(0, len(data) - 1):
        line = data[i].split(',')  # deleted replace(...)
        current_prices_dataset_list = [torch.tensor(float(value)) for value in line]
        prices_dataset_list.append(torch.stack(current_prices_dataset_list))
    prices_dataset = torch.stack(prices_dataset_list)
    return prices_dataset
