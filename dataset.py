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

