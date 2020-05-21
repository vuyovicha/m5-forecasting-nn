import torch
from torch.utils.data import Dataset


class DatasetTimeSeries(Dataset):
    def __init__(self, train_dataset, val_dataset, categories, device):
        self.train_dataset = [torch.tensor(train_dataset[i]) for i in range(len(train_dataset))]
        self.val_dataset = [torch.tensor(val_dataset[i]) for i in range(len(val_dataset))]
        self.categories = categories
        self.device = device

    def __getitem__(self, index):
        return self.train_dataset[index].to(self.device), \
               self.val_dataset[index].to(self.device), \
               index,  \
               self.categories[index]

    def __len__(self):
        return len(self.train_dataset)


class ClassifierDataset(Dataset):
    def __init__(self, classification_dataset, device):
        self.series_index = [torch.tensor(classification_dataset[i, 0]) for i in range(len(classification_dataset[:, 0]))]
        self.day_index = [torch.tensor(classification_dataset[i, 1]) for i in range(len(classification_dataset[:, 1]))]
        self.numerical_data = [torch.tensor(classification_dataset[i, 2:8], dtype=torch.float32) for i in range(len(classification_dataset[:, 2:8]))]
        self.categorical_data = [torch.tensor(classification_dataset[i, 8:20], dtype=torch.int64) for i in range(len(classification_dataset[:, 8:20]))]
        self.target = [torch.tensor(classification_dataset[i, 20]) for i in range(len(classification_dataset[:, 20]))]
        self.device = device

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.series_index[index].to(self.device), \
               self.day_index[index].to(self.device), \
               self.numerical_data[index].to(self.device),\
               self.categorical_data[index].to(self.device), \
               self.target[index].to(self.device), \
               index


class ClassifierValDatset(Dataset):
    def __init__(self, val_data, device):
        self.series_index = [torch.tensor(val_data[i, 0]) for i in range(len(val_data[:, 0]))]
        self.day_index = [torch.tensor(val_data[i, 1]) for i in range(len(val_data[:, 1]))]
        self.numerical_data = [torch.tensor(val_data[i, 2:8], dtype=torch.float32) for i in range(len(val_data[:, 2:8]))]
        self.categorical_data = [torch.tensor(val_data[i, 8:20], dtype=torch.int64) for i in range(len(val_data[:, 8:20]))]
        self.device = device

    def __len__(self):
        return len(self.series_index)

    def __getitem__(self, index):
        return self.series_index[index].to(self.device), \
               self.day_index[index].to(self.device), \
               self.numerical_data[index].to(self.device),\
               self.categorical_data[index].to(self.device), \
               index


