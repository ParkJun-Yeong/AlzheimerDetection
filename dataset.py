import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DementiaDataset(Dataset):
    def __init__(self, is_train):
        super(DementiaDataset, self).__init__()

        self.is_train = is_train
        self.base_path = os.path.join(os.getcwd(), 'dataset')
        self.corpus = pd.read_csv(os.path.join(self.base_path, "corpus.csv"))

        # self.control_path = os.path.join(self.base_path, 'control')
        # self.dementia_path = os.path.join(self.base_path, 'dementia')
        # self.control_files = os.listdir(self.control_path)
        # self.dementia_files = os.listdir(self.dementia_path)

        self.dataset = self.corpus["sentence"].astype("string")
        self.label = self.corpus["label"]

        self.train_dataset , self.test_dataset, self.train_label, self.test_label = self.split_test()

    def __len__(self):
        if self.is_train:
            return len(self.train_dataset)
        else:
            return len(self.test_dataset)

    def __getitem__(self, idx):
        # cut_line = len(self.control_files)
        #
        # if idx <= cut_line:
        #     file_path = os.path.join(self.control_path, self.dataset[idx])
        #     label = 0
        # else:
        #     file_path = os.path.join(self.dementia_path, self.dataset[idx])
        #     label = 1
        if self.is_train:
            data = self.train_dataset[idx]
            label = self.train_label[idx]
        else:
            data = self.test_dataset[idx]
            label = self.test_label[idx]

        return data, label

    def split_test(self):
        num_dataset = len(self.dataset)

        x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.label, test_size=0.1,
                                                            shuffle=True, stratify=self.label, random_state=1024)

        return x_train, x_test, y_train, y_test

