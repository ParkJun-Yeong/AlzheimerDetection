import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from embedding import Embedding
import pickle
import torch
import numpy as np


class DementiaDataset(Dataset):
    def __init__(self, train=False, valid=False, test=False):
        super(DementiaDataset, self).__init__()

        self.train = train
        self.valid = valid
        self.test = test

        self.base_path = './dataset/xml'
        self.corpus = pd.read_csv(os.path.join(self.base_path, "corpus.csv"))
        # self.transform = transforms.Compose([transforms.ToTensor()])

        # file_num이 동일한 것끼리
        self.dataset = [self.corpus.loc[self.corpus['file_num'] == i, :] for i in range(552)]
        # embedded_sent = Embedding.bert_embedding(self.dataset[i])


        # # 피클 파일 없을때만 실행
        # self.corpus_dict = [{'who': self.dataset[i].loc[:, "who"].values.tolist(),
        #                      'sentence': Embedding.bert_embedding(self.dataset[i].loc[:, 'sentence'].values.tolist())} for i in range(552)]
        #
        # with open("./corpus_dict.pkl", 'wb') as f:
        #     pickle.dump(self.corpus_dict, f)

        with open(os.path.join(self.base_path, "corpus_dict.pkl"), 'rb') as f:
            self.corpus_dict = pickle.load(f)

        # self.dataset = self.database
        self.label = [1]*309 + [0]*243

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
            data = self.x_train[idx]
            label = self.y_train[idx]
        elif self.valid:
            data = self.x_valid[idx]
            label = self.y_valid[idx]
        elif self.test:
            data = self.x_test[idx]
            label = self.y_test[idx]

        # data.update(label=label)
        # label = torch.tensor(label)

        return data, label

    # train(7), valid(2), test(1)
    def split_test(self):
        # num_dataset = len(self.dataset)

        x_train, x_test, y_train, y_test = train_test_split(self.corpus_dict, self.label, test_size=0.1,
                                                            shuffle=True, stratify=self.label, random_state=1024)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2,
                                                              shuffle=True, stratify=y_train, random_state=1024)

        return x_train, x_valid, x_test, y_train, y_valid, y_test


def collate_fn(data):
    """
    We should build a custom collate_fn rather than using default collate_fn,
    as the size of every sentence is different and merging sequences (including padding)
    is not supported in default.
    Args:
        data: list of tuple (training sequence, label)
    Return:
        padded_seq - Padded Sequence, tensor of shape (batch_size, padded_length)
        length - Original length of each sequence(without padding), tensor of shape(batch_size)
        label - tensor of shape (batch_size)
    """

    #sorting is important for usage pack padded sequence (used in model). It should be in decreasing order.
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    dialogue, label = zip(*data)
    # length = [len(seq) for seq in dialogue]
    # padded_seq = torch.zeros(len(dialogue), max(length)).long()
    # for i, seq in enumerate(dialogue):
    #     end = length[i]
    #     padded_seq[i,:end] = seq
    return dialogue, label

def move(self, d: dict, device) -> dict:
    for k in d:
        if isinstance(d[k], dict):
            d[k] = self.move(d[k])
        # elif isinstance(d[k], (Tensor)):
        #     d[k] = d[k].to(device=device, non_blocking=True)
    return d
