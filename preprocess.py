import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd



import torch.nn

class DementiaDataset(Dataset):
    def __init__(self):
        super(DementiaDataset, self).__init__()
        self.base_path = os.path.join(os.getcwd(), 'dataset')
        self.control_path = os.path.join(self.base_path, 'control')
        self.dementia_path = os.path.join(self.base_path, 'dementia')

        self.control_files = os.listdir(self.control_path)
        self.dementia_files = os.listdir(self.dementia_path)
        self.dataset = self.control_files + self.dementia_files


    def __len__(self):
        return len(self.control_files) + len(self.dementia_files)

    def __getitem__(self, idx):
        cutline = len(self.control_files)

        if idx <= cutline:
            file_path = os.path.join(self.control_path, self.dataset[idx])
            label = 0
        else:
            file_path = os.path.join(self.dementia_path, self.dataset[idx])
            label = 1

        file = pd.read_csv(file_path, delimiter='\n')

        return file, label


class Preprocess:
    def __init__(self):

        self.max_seq_len = 100          # 언급 없음

        self.dementia = []
        self.control = []

    def loader(self):

    def padding(self):
