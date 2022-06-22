import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DementiaDataset(Dataset):
    def __init__(self, train=False, valid=False, test=False):
        super(DementiaDataset, self).__init__()

        self.train = train
        self.valid = valid
        self.test = test

        self.base_path = os.path.join(os.getcwd(), 'dataset')
        self.corpus = pd.read_csv(os.path.join(self.base_path, "corpus.csv"))

        # sentence 뿐만 아니라 라벨 열 제외 전체 열 데이터를 dataset 으로 함
        self.dataset = self.corpus["sentence"].astype("string")
        self.label = self.corpus["label"]

        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = self.split_test()

        # self.control_path = os.path.join(self.base_path, 'control')
        # self.dementia_path = os.path.join(self.base_path, 'dementia')
        # self.control_files = os.listdir(self.control_path)
        # self.dementia_files = os.listdir(self.dementia_path)

    def __len__(self):
        if self.train:
            return len(self.x_train)
        elif self.valid:
            return len(self.x_valid)
        elif self.test:
            return len(self.x_test)

    def __getitem__(self, idx):
        # cut_line = len(self.control_files)
        #
        # if idx <= cut_line:
        #     file_path = os.path.join(self.control_path, self.dataset[idx])
        #     label = 0
        # else:
        #     file_path = os.path.join(self.dementia_path, self.dataset[idx])
        #     label = 1

        if self.train:
            data = self.x_train.iloc[idx]
            label = self.y_train.iloc[idx]
        elif self.valid:
            data = self.x_valid.iloc[idx]
            label = self.y_valid.iloc[idx]
        elif self.test:
            data = self.x_test.iloc[idx]
            label = self.y_test.iloc[idx]

        return data, label

    # train(7), valid(2), test(1)
    def split_test(self):
        # num_dataset = len(self.dataset)

        x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.label, test_size=0.1,
                                                            shuffle=True, stratify=self.label, random_state=1024)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2,
                                                              shuffle=True, stratify=y_train, random_state=1024)

        return x_train, x_valid, x_test, y_train, y_valid, y_test
