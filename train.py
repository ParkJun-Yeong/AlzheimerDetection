import torch
import torch.nn as nn
import os
import sys
import embedding
import model


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    lr = 1e-3
    batch_size = 100        # 임의 지정. 바꾸기.
    dropout_rate = 0.3      # 논문 언급 없음.
    weight_decay = 2e-5
    embedding_size = 100    # 여러 차원으로 실험해보기
    max_seq_length = 100    # 논문 언급 없음.
    seed = 1024
    num_classes = 2

    # 10-fold cross validation 적용

    # Embedding

    """
    CNN + Attention
    """

    """
    BiGRU + Attention
    """

    """
    concat + Softmax
    """