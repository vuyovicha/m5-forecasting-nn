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
        self.numerical_data = [torch.tensor(classification_dataset[i, 0:5], dtype=torch.float32) for i in range(len(classification_dataset[:, 0:5]))]
        self.categorical_data = [torch.tensor(classification_dataset[i, 5:12], dtype=torch.int64) for i in range(len(classification_dataset[:, 5:12]))]
        self.target = [torch.tensor(classification_dataset[i, 12]) for i in range(len(classification_dataset[:, 12]))]
        self.device = device

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.numerical_data[index].to(self.device),\
               self.categorical_data[index].to(self.device), \
               self.target[index].to(self.device), \
               index


class ClassifierValDatset(Dataset):
    def __init__(self, val_data, device):
        self.numerical_data = [torch.tensor(val_data[0:5], dtype=torch.float32) for i in range(1)]
        self.categorical_data = [torch.tensor(val_data[5:], dtype=torch.int64) for i in range(1)]
        self.device = device

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.numerical_data[index].to(self.device),\
               self.categorical_data[index].to(self.device), \
               index


