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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for i, (X, y) in tqdm(enumerate(dataloader), desc="Train..."):
        # Prediction and Loss
        y = y.to(device)
        embedded_x = Embedding.bert_embedding(X).to(device)
        pred = model(embedded_x)
        pred = torch.squeeze(pred, dim=-1)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 15 == 0:
            saved_model_dir = "/home/juny/AlzheimerModel"
            # saved_model_dir = "./saved_model"
            now = datetime.now()
            torch.save(model, os.path.join(saved_model_dir, "saved_model" + now.strftime("%Y-%m-%d-%H-%M") + ".pt"))
            loss, current = loss.item(), i * len(X)
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

    merge_mode = "concat"
    embedding = 'bert'  # choose: bert, word2vec, glove, torch

    learning_rate = 1e-3
    batch_size = 64        # 임의 지정. 바꾸기.
    epochs = 10
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

    model = Model(batch_size=batch_size, max_seq_length=max_seq_length, dropout_rate=dropout_rate,
                  vocab_size=vocab_size, merge_mode=merge_mode, embedding_size=embedding_size).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # train_feature, train_labels = next(iter(train_dataloader))
    # print(train_feature)
    # print(train_labels)

    print("========================[[Train]]========================")
    print()

    for _ in range(epochs):
        train_loop(dataloader=train_dataloader, model=model,
                   loss_fn=loss_fn, optimizer=optimizer)

    print("========================[[Validation]]========================")
    print()
    train_loop(dataloader=valid_dataloader, model=model,
               loss_fn=loss_fn, optimizer=optimizer)
