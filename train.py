import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Model
import preprocess


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        # Prediction and Loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}")


def cross_validation(dataloader, model, loss_fn):

    return


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embedding = 'bert'                              # choose: bert, word2vec, glove, torch

    learning_rate = 1e-3
    batch_size = 64        # 임의 지정. 바꾸기.
    epochs = 5
    dropout_rate = 0.3      # 논문 언급 없음.
    weight_decay = 2e-5
    embedding_size = 100    # 여러 차원으로 실험해보기
    max_seq_length = 100    # 논문 언급 없음.
    seed = 1024
    merge_mode = "concat"

    vocab_size = 1751
    num_classes = 2


    # Dataloader (Test, validation도 만들기)
    dataset = preprocess.DementiaDataset()
    train_dataloader = DataLoader(dataset, batch_size=batch_size)

    model = Model(max_seq_length, dropout_rate, vocab_size, merge_mode, embedding_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loop(train_dataloader, model, loss_fn, optimizer)

    # 10-fold cross validation 적용

    # Embedding

