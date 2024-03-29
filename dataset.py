import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import trange
from embedding import Embedding
from preprocess import Preprocess
import pickle
import torch
import numpy as np
from dataclasses import dataclass   # 구조체

"""
** DataStruct: 한 파일에 담긴 대화 데이터, 여러 Section으로 구성
** Section: INV 발화 하나와 여러 PAR 발화의 구성

1. Section 개수는 파일마다 유동적이다.
2. 한 Section에는 하나의 INV와 여러 PAR이 들어있다.
3. DataStruct 내에서는 Section들의 순서를 고려한다.
4. 각 DataStruct는 독립적이다. (다른 파일이므로)
"""

# Section 별로 나누어서 학습할 때
@dataclass
class DataStruct:
    def __init__(self):
        self.sections = []


@dataclass
class Section:
    def __init__(self):
        self.inv = None
        self.par = []
        self.next_uttr = None


# Section을 나누지 않고 학습할 때
@dataclass()
class Dialogue:
    def __init__(self):
        self.dialogue = []
        self.who = []


# cross validation 할 거기 때문에 valid 데이터셋은 따로 안둠
class DementiaDataset(Dataset):
    def __init__(self, is_tr=False, is_ts=False):
        super(DementiaDataset, self).__init__()
        if (not is_tr) & (not is_ts):
            print("!!!!!!!!!! Select is_tr or is_ts !!!!!!!!!!")
            exit()

        self.is_tr = is_tr          # return train dataset
        self.is_ts = is_ts          # return test dataset

        self.CONFIG_ = {"REALTIME_EMBED_": False,
                        "FIRST_EMBED_": True,
                        "TYPE_OF_EMBED_": "glove300"}  # bert, glove100, glove300

        self.base_path = './dataset'

        """
        직접 임베딩
        """
        if self.CONFIG_["FIRST_EMBED_"]:
            self.corpus = pd.read_csv(os.path.join(self.base_path, "corpus.csv"))
            self.corpus = Preprocess.fill_inv(self.corpus).drop(['index'], 1)         # index 항 제거 (pandas method로 해결 가능)
            self.sentences = self.corpus["setence"].astype("string")

            if self.CONFIG_["TYPE_OF_EMBED_"] == "bert":
                self.corpus_dict = [{'who': self.corpus.loc[:, "who"].values.tolist(),
                                                          'sentence': Embedding.bert_embedding(self.corpus[i].loc[:, 'sentence'].values.tolist())} for i in range(552)]
                with open("./dataset/embed_by_bert.pkl", 'wb') as f:
                    pickle.dump(self.corpus_dict, f)


            if self.CONFIG_["TYPE_OF_EMBED_"] == "glove300":
                self.corpus_dict = [{'who': self.corpus.loc[:, "who"].values.tolist(),
                                     'sentence': Embedding.glove_embedding(self.corpus[i].loc[:, "sentence"].values.tolist())} for i in range(552)]

                with open("./dataset/embed_by_glove300.pkl", 'wb') as f:
                    pickle.dump(self.corpus_dict, f)

        """
        임베딩 불러와서
        """
        if self.CONFIG_["TYPE_OF_EMBED_"] == "bert":
            with open(os.path.join(self.base_path, "embed_by_bert.pkl"), 'rb') as f:
                self.corpus = pickle.load(f)

        if self.CONFIG_["TYPE_OF_EMBED_"] == "glove300":
            with open(os.path.join(self.base_path, "embed_by_glove300.pkl"), 'rb') as f:
                self.corpus = pickle.load(f)

        self.corpus = Preprocess.fill_inv_from_dict(self.corpus)
        self.dataset = []

        # # Section 분리
        # self.get_struct()
        self.get_struct_from_dict()

        # Section 비분리
        # self.get_dialogue()

        # len_true = len(self.corpus.loc[self.corpus['label'] == 1]['file_num'].unique())         # 309
        # len_false = len(self.corpus.loc[self.corpus['label'] == 0]['file_num'].unique())


        # self.label = [1] * len_true + [0] * len_false

        self.label = [1] * 309 + [0] * 243
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_dataset()

    def __len__(self):
        if self.is_tr:
            return len(self.x_train)
        elif self.is_ts:
            return len(self.x_test)

    # 반환 단위: DataStruct (containing multiple Sections)
    def __getitem__(self, idx):
        if self.is_tr:
            return self.x_train[idx], self.y_train[idx]
        elif self.is_ts:
            return self.x_test[idx], self.y_test[idx]

    """
    corpus를 동일 파일로 구분 & 각 파일을 여러 section으로 나눔
    데이터를 구조체로 정리해 구성하는 메소드
    """
    def get_struct(self):
        for i in trange(552):
            struct = DataStruct()
            try:  # 마지막이 inv로 끝날 경우 sec에 utter가 남아있음. 초기화 해주기
                del sec
            except UnboundLocalError:
                pass
                # print("let's start to make structure **^^**")

            file = self.corpus.loc[self.corpus['file_num'] == i, :]  # 동일 파일 행만 추출

            for j in range(len(file)):
                uttr = file.iloc[j]['sentence']
                # print(i, j)
                if file.iloc[j]['who'] == 'INV':
                    try:
                        sec.next_uttr = uttr
                        struct.sections.append(sec)
                        # sec.par = []
                    except UnboundLocalError:
                        pass
                        # print("no section: ", i, j)
                        # print("no section")
                    sec = Section()  # 새로운 세션 생성
                    sec.inv = uttr

                if file.iloc[j]['who'] == 'PAR':
                    sec.par.append(uttr)
            self.dataset.append(struct)
            
    def get_struct_from_dict(self):
        for i in trange(552):
            struct = DataStruct()

            try:  # 마지막이 inv로 끝날 경우 sec에 utter가 남아있음. 초기화 해주기
                del sec
            except UnboundLocalError:
                pass
                # print("let's start to make structure **^^**")

            file = self.corpus[i]  # 동일 파일만 추출

            for j in range(len(file['who'])):
                uttr = file['sentence'][j].unsqueeze(0)
                # print(i, j)
                if file['who'][j] == 'INV':
                    try:
                        sec.next_uttr = uttr
                        struct.sections.append(sec)
                        # sec.par = []
                    except UnboundLocalError:
                        pass
                        # print("no section: ", i, j)
                        # print("no section")
                    sec = Section()  # 새로운 세션 생성
                    sec.inv = uttr

                if file['who'][j] == 'PAR':
                    if len(sec.par) == 0:
                        sec.par = uttr
                    else:
                        torch.concat((sec.par, uttr))
            self.dataset.append(struct)

        print("STRUCTURING COMPLETE")
    """
    같은 파일의 발화(sentence)와 발화자(who)를 빼서 Dialogue 구조체로 정리하는 메소드
    Section으로 분리하지 않을 경우만 사용
    """
    def get_dialogue(self):
        for i in trange(552):
            file = self.corpus.loc[self.corpus['file_num'] == i, :]  # 동일 파일 행만 추출
            diag = Dialogue()

            for j in range(len(file)):
                uttr = file.iloc[j]['sentence']
                who = file.iloc[j]['who']

                diag.dialogue.append(uttr)
                diag.who.append(who)

            self.dataset.append(diag)


    def split_dataset(self):
        x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.label, test_size=0.1, stratify=self.label
                                                            , shuffle=True, random_state=1024)
        return x_train, x_test, y_train, y_test

"""
get item에서 보내진 데이터를 zip으로 분리해서 내보냄
"""
def collate_fn(data):
    X, label = zip(*data)
    return X, label

from torch.utils.data import DataLoader
if __name__ == "__main__":
    data = DementiaDataset(is_tr=True)
    dataloader = DataLoader(data, shuffle=True, batch_size=3, collate_fn=collate_fn)

