import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime
from tqdm import tqdm

from preprocess import Preprocess
from dataset import DementiaDataset, collate_fn
from embedding import Embedding
from AlzhBERT import AlzhBERT

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_loop(dataloader, model, loss_fn, optimizer, epochs):

    train_dataloader = dataloader["train"]
    valid_dataloader = dataloader["valid"]

    batch_size = 32
    size = len(train_dataloader.dataset)
    writer = SummaryWriter()

    # loss_history = []
    # train_loss_history = []
    # valid_loss_history = []

    for epoch in range(epochs):
        print("======== epoch ", epoch, "==========")

        for i, (X, y) in tqdm(enumerate(train_dataloader), desc="Train..."):
            model.train()
            # Prediction and Loss
            y = torch.tensor(y, dtype=int)
            y = y.to(device)
            # embedded_x = Embedding.bert_embedding(X["sentence"]).to(device)

            enc_loss, dec_loss = model(X, y)

            # (cls_out, decoder_out, decoder_tgt)
            # pred 모드 따로 지정
            # pred = model(X, y)
            # pred = torch.squeeze(pred, dim=-1)
            # loss = loss_fn(pred, y)

            # loss_history.append(loss.data)
            writer.add_scalar("Enc Loss/train", enc_loss, epoch)
            writer.add_scalar("Dec Loss/train", dec_loss, epoch)

            # Backpropagation
            optimizer.zero_grad()
            enc_loss.backward(retain_graph=True)
            dec_loss.backward()
            optimizer.step()

            if i % 5 == 0:
                if device is "cuda":
                    saved_model_dir = "/home/juny/AlzheimerModel"
                else:
                    saved_model_dir = "./saved_model"
                # saved_model_dir = "./saved_model"
                now = datetime.now()
                torch.save(model, os.path.join(saved_model_dir, "saved_model" + now.strftime("%Y-%m-%d-%H-%M") + ".pt"))
                encloss, decloss, current = enc_loss.item(), dec_loss.item(), i * len(X)
                print(f"enc loss: {encloss:>7f} dec loss: {decloss:>7f} [{current:>5d}/{size:>5d}")

        enc_valid_loss, dec_valid_loss = validation_loop(valid_dataloader, model, loss_fn, epoch)
        writer.add_scalar("Enc Loss/valid", enc_valid_loss, epoch)
        writer.add_scalar("Dec Loss/valid", dec_valid_loss, epoch)

    writer.flush()
    writer.close()


def validation_loop(dataloader, model, loss_fn, epoch):
    writer = SummaryWriter()
    model.eval()
    enc_loss_sum = 0
    dec_loss_sum = 0

    # loss_history = []
    # val_loss_history = []
    with torch.no_grad():
        val_loss = 0.0
        for i, (X, y) in enumerate(dataloader):
            y = torch.tensor(y, dtype=int)
            y = y.to(device)
            # embedded_x = Embedding.bert_embedding(X).to(device)

            enc_loss, dec_loss = model(X, y)
            # pred = torch.squeeze(pred, dim=-1)

            enc_loss_sum += enc_loss
            dec_loss_sum += dec_loss

            # loss = loss_fn(pred, y)
            # loss_history.append(loss)
            # val_loss += loss.data

        # val_loss_history.append(val_loss)
    return enc_loss_sum, dec_loss_sum

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

    # merge_mode = "concat"
    embedding = 'bert'  # choose: bert, word2vec, glove, torch

    learning_rate = 1e-3
    batch_size = 32        # 임의 지정. 바꾸기.
    epochs = 10
    dropout_rate = 0.1      # 논문 언급 없음.
    weight_decay = 2e-5
    max_seq_length = 30    # 논문 언급 없음.
    seed = 1024
    num_classes = 2

    pre = Preprocess()
    if embedding == 'bert':
        embedding_size = 768
        vocab_size = None
    else:
        embedding_size = 100  # 여러 차원으로 실험해보기
        vocab, vocab_size = pre.tokenize()


    # Dataloader
    train_dataset = DementiaDataset(train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)

    for i, (X, label) in enumerate(train_dataloader):
        print(i, ':', X, label)

    valid_dataset = DementiaDataset(valid=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = DementiaDataset(test=True)
    test_dataloader = DataLoader(test_dataset, shuffle=True, collate_fn=collate_fn)
    #
    model = AlzhBERT(pred=False).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    # train_feature, train_labels = next(iter(train_dataloader))
    # print(train_feature)
    # print(train_labels)

    print("========================[[Train]]========================")
    print()

    train_loop(dataloader={"train": train_dataloader, "valid": valid_dataloader}, model=model,
               loss_fn=loss_fn, optimizer=optimizer, epochs=epochs)

    # print("========================[[Validation]]========================")
    # print()
    # train_loop(dataloader=valid_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
