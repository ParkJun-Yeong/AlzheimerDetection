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
print("device: ", device)


def train_loop(dataloader, model, loss_fn, optimizer, epochs):
    dataloader = dataloader["train"]
    size = len(train_dataloader.dataset)
    writer = SummaryWriter()

    for epoch in range(epochs):
        print("======== epoch ", epoch, "==========\n")
        for i, (Xs, ys) in tqdm(enumerate(train_dataloader), desc="Train..."):
            X_folds,  y_folds = cross_validation(10, Xs, ys)
            model.train()

            for X, y in zip(X_folds['train'], y_folds['train']):                    # Xf는 DataStruct의 리스트임
                print("<Check Data>")
                print("X 0: ", X[0])
                print("label 0: ", y[0])

                # Prediction and Loss
                X = torch.tensor(X).to(device)
                y = torch.tensor(y, dtype=int).to(device)

                enc_loss, dec_loss, sample_num = model(X, y)

            # loss_history.append(loss.data)
            writer.add_scalar("Total Enc Loss/train", enc_loss, epoch)
            writer.add_scalar("Total Dec Loss/train", dec_loss, epoch)
            mean_enc_train = enc_loss/float(sample_num)
            mean_dec_train = dec_loss/float(sample_num)
            writer.add_scalar("Mean Enc Loss/train", mean_enc_train, epoch)
            writer.add_scalar("Mean Dec Loss/train", mean_dec_train, epoch)

            # Backpropagation
            optimizer.zero_grad()
            enc_loss.backward(retain_graph=True)
            dec_loss.backward()
            optimizer.step()

            if i % 5 == 0:
                if device == "cuda":
                    saved_model_dir = "/home/juny/AlzheimerModel"
                else:
                    saved_model_dir = "./saved_model"
                # saved_model_dir = "./saved_model"
                now = datetime.now()
                torch.save(model.state_dict(), os.path.join(saved_model_dir, "saved_model" + now.strftime("%Y-%m-%d-%H-%M") + ".pt"))
                encloss, decloss, current = mean_enc_train.item(), mean_dec_train.item(), i * len(X)
                print(f"enc loss: {encloss:>7f} dec loss: {decloss:>7f} [{current:>5d}/{size:>5d}")

        # enc_valid_loss, dec_valid_loss, valid_acc, sample_num = validation_loop(valid_dataloader, model, loss_fn, epoch)
        # writer.add_scalar("Total Enc Loss/valid", enc_valid_loss, epoch)
        # mean_enc_valid = enc_valid_loss/sample_num
        # writer.add_scalar("Mean Enc Loss/valid", mean_enc_valid, epoch)
        #
        # writer.add_scalar("Total Dec Loss/valid", dec_valid_loss, epoch)
        # mean_dec_valid = dec_valid_loss/sample_num
        # writer.add_scalar("Mean Dec Loss/valid", mean_dec_valid, epoch)
        # writer.add_scalar("Mean Accuracy/valid", valid_acc, epoch)
        #
        # print(f"Valid enc loss: {mean_enc_valid:>7f} dec loss: {mean_dec_valid:>7f} acc: {valid_acc:>7f} [{current:>5d}/{size:>5d}")

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

            enc_loss, dec_loss, acc, sample_num = model(X, y, valid=True)
            # pred = torch.squeeze(pred, dim=-1)

            enc_loss_sum += enc_loss
            dec_loss_sum += dec_loss

            # loss = loss_fn(pred, y)
            # loss_history.append(loss)
            # val_loss += loss.data

        # val_loss_history.append(val_loss)
    return enc_loss_sum, dec_loss_sum, acc, float(sample_num)


"""
cross validation 가능하도록 데이터셋을 나눠줌
1. test데이터가 있다면 반드시 train에서 분리 후 사용
2. k_fold: 몇 개의 fold로 나눌지
3. 다른 곳에서도 사용할수도 있으니 staticmethod 지정
4. 전체 train 데이터를 나누는게 아니라 train dataloader를 통해 나온 batch를 k_folds로 나누는 것 (전체 데이터를 나눠야한다면 코드 수정필요)

** 리턴값: data, label 딕셔너리
data['train'][0] -> 첫번째 fold 학습데이터들
label['train'][0] -> 첫번째 fold 라벨들
"""
def cross_validation(k, batch_X, batch_y):
    length = len(batch_X)
    n = int(length / k)
    folds = []
    folds_y = []
    data = {'train': [], 'valid': []}      # 데이터
    label = {'train': [], 'valid': []}      # 라벨

    print(n)

    i = 0

    for i in range(k):
        if i == (k-1):
            folds.append(batch_X[n*i:])          # 마지막은 끝까지
            folds_y.append(batch_y[n*i:])
        else:
            folds.append(batch_X[n*i:n*(i+1)])
            folds_y.append(batch_y[n*i:n*(i+1)])

    for i in range(k):
        data['valid'].append(folds[i])
        data['train'].append([f for f in folds if f not in folds[i]])      # valid 제외한 것

        label['valid'].append(folds_y[i])
        label['train'].append([f for f in folds if f not in folds_y[i]])

    return data, label

if __name__ == "__main__":

    # merge_mode = "concat"
    embedding = 'bert'  # choose: bert, word2vec, glove, torch

    learning_rate = 1e-3
    batch_size = 64        # 임의 지정. 바꾸기.
    epochs = 70
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
        embedding_size = 100  # 여러 차원으로W 실험해보기
        vocab, vocab_size = pre.tokenize()


    # Dataloader
    train_dataset = DementiaDataset(is_tr=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # for i, (X, label) in enumerate(train_dataloader):
    #     print(i, ':', X, label)


    # test_dataset = DementiaDataset(test=True)
    # test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn)

    model = AlzhBERT(max_token_num, max_seq_len, num_heads=6).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    # train_feature, train_labels = next(iter(train_dataloader))
    # print(train_feature)
    # print(train_labels)

    print("========================[[Train]]========================\n")
    print()

    train_loop(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs)

    # print("========================[[Validation]]========================")
    # print()
    # train_loop(dataloader=valid_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
