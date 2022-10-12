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
from utils import get_y_dec_list, get_y_enc_list

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
torch.autograd.set_detect_anomaly(True)

def train_loop(dataloader, model, loss_fn, optimizer, epochs):
    # dataloader = dataloader["train"]
    size = len(dataloader.dataset)
    writer = SummaryWriter()

    for epoch in range(epochs):
        enc_loss_hist = []
        dec_loss_hist = []
        accuracy = []

        valid_enc_hist = []
        valid_dec_hist = []
        valid_accuracy = []

        print("======== epoch ", epoch, "==========\n")
        for i, (Xs, ys) in tqdm(enumerate(dataloader), desc="Train..."):
            X_folds,  y_folds = cross_validation(2, Xs, ys)
            model.train()

            for X, y in zip(X_folds['train'], y_folds['train']):                    # Xf는 DataStruct의 리스트임
                # print("<Check Data>")
                # print("X 0: ", X[0])
                # print("label 0: ", y[0])

                # Prediction and Loss
                # X = batch_to_tensor(X)
                # X = torch.tensor(X).to(device)
                y = torch.tensor(y, dtype=torch.float32).to(device)
                y_dec = get_y_dec_list(X)
                y_enc = get_y_enc_list(X, y)
                # y_dec = torch.tensor(y_dec, requires_grad=True)
                y_enc = torch.tensor(y_enc, requires_grad=True)


                enc_preds, dec_preds = model(X)

                # for i in range(len(dec_preds)):
                #     y_dec = X[1].sections[i].next_uttr.to(device)
                #     dec_pred = dec_preds[i].to(device)
                #     dec_loss = loss_fn(y_dec, dec_pred.requires_grad_(True))
                #
                #     try:
                #         dec_optimizer.zero_grad()
                #         dec_loss.backward()
                #         dec_optimizer.step()
                #     except RuntimeError:
                #         print(i, " 번째 iteration of X[1]")
                #         continue

                enc_preds = torch.stack(enc_preds)
                # dec_preds = torch.stack(dec_preds)

                enc_loss = loss_fn(y_enc, enc_preds)

                dec_losses = []
                for i in range(len(dec_preds)):
                    dec_losses.append(loss_fn(y_dec[i], dec_preds[i]))

                dec_losses = torch.stack(dec_losses)
                dec_loss = torch.mean(dec_losses)

                cls_out = [1 if enc >= 0.5 else 0 for enc in enc_preds]
                accuracy.extend(cls_out)


                # Backpropagation
                # enc_optimizer.zero_grad()
                # enc_loss = torch.mean(enc_losses)
                # dec_loss = torch.mean(dec_losses)

                # dec_optimizer.zero_grad()
                optimizer.zero_grad()
                enc_loss.backward(retain_graph=True)
                dec_loss.backward()
                optimizer.step()

                # dec_loss.backward()
                # dec_optimizer.step()

                enc_loss_hist.append(enc_loss.item())
                dec_loss_hist.append(dec_loss.item())

                enc_valid, dec_valid, acc_valid = cross_validation_loop(X_folds["valid"], y_folds["valid"], model, loss_fn, epoch)
                valid_enc_hist.append(enc_valid)
                valid_dec_hist.append(dec_valid)
                valid_accuracy.extend(acc_valid)

        enc_loss_save = torch.mean(torch.tensor(enc_loss_hist))
        dec_loss_save = torch.mean(torch.tensor(dec_loss_hist))
        accuracy_save = torch.mean(torch.tensor(accuracy, dtype=torch.float32))

        valid_enc_save = torch.mean(torch.tensor(valid_enc_hist))
        valid_dec_save = torch.mean(torch.tensor(valid_dec_hist))
        valid_acc_save = torch.mean(torch.tensor(valid_accuracy, dtype=torch.float32))

        writer.add_scalar("Avg Enc Loss/train", enc_loss_save, epoch)
        writer.add_scalar("Avg Dec Loss/train", dec_loss_save, epoch)
        writer.add_scalar("Avg Accuracy/train", accuracy_save)

        writer.add_scalar("Avg Enc Loss/valid", valid_enc_save, epoch)
        writer.add_scalar("Avg Enc Loss/valid", valid_dec_save, epoch)
        writer.add_scalar("Avg Enc Loss/valid", valid_acc_save, epoch)


        if device == "cuda":
            saved_model_dir = "/home/juny/AlzheimerModel/checkpoint"
        else:
            saved_model_dir = "./saved_model"

        now = datetime.now()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': [enc_loss_save, dec_loss_save],
        }, os.path.join(saved_model_dir,
                        now.strftime("%Y-%m-%d-%H-%M") + "-e" + str(epoch) + ".pt"))

        # torch.save(model.state_dict(), os.path.join(saved_model_dir, "saved_model" + now.strftime("%Y-%m-%d-%H-%M") + ".pt"))
        encloss, decloss, current = enc_loss_save, dec_loss_save, i * len(X)
        print(f"enc loss: {encloss:>7f} dec loss: {decloss:>7f} [{current:>5d}/{size:>5d}")
        encloss, decloss, current = valid_enc_save, valid_dec_save, i * len(X)
        print(f"valid enc loss: {encloss:>7f} valid dec loss: {decloss:>7f} [{current:>5d}/{size:>5d}")

    writer.flush()
    writer.close()


def cross_validation_loop(X_fold, y_fold, model, loss_fn, epoch):
    writer = SummaryWriter()
    model.eval()
    enc_loss_hist = []
    dec_loss_hist = []
    accuracy = []

    with torch.no_grad():
        for X, y in tqdm(zip(X_fold, y_fold), desc="[[validation loop...]]"):

            y = torch.tensor(y, dtype=torch.float32).to(device)
            y_enc = get_y_enc_list(X, y)
            y_dec = get_y_dec_list(X)

            enc_preds, dec_preds = model(X)

            y_enc = torch.tensor(y_enc, device=device)
            enc_preds = torch.stack(enc_preds)
            enc_loss = loss_fn(y_enc, enc_preds)

            dec_losses = []
            for i in range(len(y_dec)):
                loss = loss_fn(y_dec[i], dec_preds[i])
                dec_losses.append(loss)

            dec_losses = torch.stack(dec_losses)
            dec_loss = torch.mean(dec_losses)

            enc_loss_hist.append(enc_loss.item())
            dec_loss_hist.append(dec_loss.item())

            cls_out = [1 if enc >= 0.5 else 0 for enc in enc_preds]

    return enc_loss, dec_loss, cls_out


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
            y = torch.tensor(y, dtype=torch.float32)
            y = y.to(device)

            enc_loss, dec_loss, acc, sample_num = model(X, y, valid=True)

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
    x_folds = None
    y_folds = None
    data = {'train': [], 'valid': []}      # 데이터
    label = {'train': [], 'valid': []}      # 라벨

    print(n)

    for i in range(k):
        if i == (k-1):
            x_folds = batch_X[n*i:]          # 마지막은 끝까지
            y_folds = batch_y[n*i:]
            data['valid'].append(x_folds)
            label['valid'].append(y_folds)

            x_folds = batch_X[:n*i]
            y_folds = batch_y[:n*i]
            data['train'].append(x_folds)
            label['train'].append(y_folds)

        else:
            x_folds = batch_X[n*i:n*(i+1)]
            y_folds = batch_y[n*i:n*(i+1)]
            data['valid'].append(x_folds)
            label['valid'].append(y_folds)

            x_folds = batch_X[:n*i] + batch_X[n*(i+1):]
            y_folds = batch_y[:n*i] + batch_y[n*(i+1):]
            data['train'].append(x_folds)
            label['train'].append(y_folds)

    return data, label


if __name__ == "__main__":

    # merge_mode = "concat"
    embedding = 'bert'  # choose: bert, word2vec, glove, torch

    learning_rate = 1e-3
    batch_size = 11        # 임의 지정. 바꾸기.
    epochs = 100
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
    train_dataset = DementiaDataset(is_tr=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # for i, (X, label) in enumerate(train_dataloader):
    #     print(i, ':', X, label)


    # test_dataset = DementiaDataset(test=True)
    # test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn)

    model = AlzhBERT(embedding_dim=embedding_size).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    # train_feature, train_labels = next(iter(train_dataloader))
    # print(train_feature)
    # print(train_labels)

    print("========================[[Train]]========================\n")
    print()

    train_loop(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs)

    # print("========================[[Validation]]========================")
    # print()
    # train_loop(dataloader=valid_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
