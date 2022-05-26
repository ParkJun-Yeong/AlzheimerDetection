import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm

from model import Model
from preprocess import Preprocess
from dataset import DementiaDataset
from embedding import Embedding


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in tqdm(enumerate(dataloader), desc="Train..."):
        # Prediction and Loss
        embedded_x = Embedding.bert_embedding(X)
        pred = model(embedded_x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 15 == 0:
            torch.save(model, os.path.join("./saved_model", datetime.now()))
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}")


# def cross_validation(dataloader, total_size, model, loss_fn, optimizer, k_fold=10):
#     train_score = pd.Series()
#     val_score = pd.Series()
#
#     total_size = total_size
#     fraction = 1/k_fold
#     seg = int(total_size * fraction)
#
#     # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset
#     # index: [trll,trlr],[vall,valr],[trrl,trrr]
#     for i in range(k_fold):
#         trll = 0
#         trlr = i * seg
#         vall = trlr
#         valr = i * seg + seg
#         trrl = valr
#         trrr = total_size
#
#         train_left_indices = list(range(trll, trlr))
#         train_right_indices = list
#
#
#
#
#
#     return

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    merge_mode = "concat"
    embedding = 'bert'  # choose: bert, word2vec, glove, torch

    learning_rate = 1e-3
    batch_size = 64        # 임의 지정. 바꾸기.
    epochs = 5
    dropout_rate = 0.3      # 논문 언급 없음.
    weight_decay = 2e-5
    max_seq_length = 30    # 논문 언급 없음.
    seed = 1024
    num_classes = 2

    pre = Preprocess()
    if embedding == 'bert':
        embedding_size = 768  # 여러 차원으로 실험해보기
        vocab_size = None
    else:
        embedding_size = 100  # 여러 차원으로 실험해보기
        vocab, vocab_size = pre.tokenize()


    # Dataloader
    train_dataset = DementiaDataset(train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = DementiaDataset(valid=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = DementiaDataset(test=True)
    test_dataloader = DataLoader(test_dataset, shuffle=True)

    model = Model(max_seq_length=max_seq_length, dropout_rate=dropout_rate, vocab_size=vocab_size, merge_mode=merge_mode, embedding_size=embedding_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("========================[[Train]]========================")
    train_loop(dataloader=train_dataloader, model=model,
               loss_fn=loss_fn, optimizer=optimizer)

    print("========================[[Validation]]========================")
    train_loop(dataloader=valid_dataloader, model=model,
               loss_fn=loss_fn, optimizer=optimizer)
