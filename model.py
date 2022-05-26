import torch
import torch.nn as nn


# 1-dimensional CNN
class CNN(nn.Module):
    def __init__(self, max_seq_len, vocab_size=None, embedding_dim=100):
        super(CNN, self).__init__()
        self.max_seq_len = max_seq_len              # sentence length
        self.embedding_dim = embedding_dim          # GloVe, Word2Vec 기준으로 범위 잡기

        # self.embbed = nn.Embedding(vocab_size, embedding_dim)
        # self.multi_size_conv = [nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(kernel_size, embedding_dim), stride=1) for kernel_size in range(1, 6)]     # first conv. layer
        # self.uni_size_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(3, embedding_dim), stride=1)
        # self.cnn = nn.ModuleList([self.multi_size_conv, self.uni_size_conv, self.uni_size_conv])

        self.conv1 = nn.ModuleList([nn.Conv1d(in_channels=self.embedding_dim, out_channels=128, kernel_size=kernel) for kernel in range(1, 6)])       # First CNN layers
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)                  # 논문 '그림 상'으로는 128이 in channel이므로 reshape 안함
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)

        self.max1d = None

    def max_pooling(self, input, kernel, stride):
        self.max1d = nn.MaxPool1d(kernel, stride)
        ret = self.max1d(input)

        return ret

    # def conv(self, input, in_channel, kernel, out_channel=128):
    #     layer = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel)
    #     ret = layer(input)
    #
    #     return ret

    def forward(self, input):
        input = input.permute(0, 2, 1)

        x = []
        for cnn in self.conv1:
            feature = cnn(input)          # Embedded Input
            feature = self.relu(feature)
            feature = self.max_pooling(feature, kernel=2, stride=2)
            x.append(feature)

        x = torch.cat(x, dim=2)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pooling(x, kernel=2, stride=2)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pooling(x, kernel=2, stride=2)

        return x


class BiRNN(nn.Module):
    def __init__(self, model, max_seq_len, dropout_rate, merge_mode=None, embedding_dim=100):
        super(BiRNN, self).__init__()
        self.model = model
        self.embedding_dim = embedding_dim
        self.hidden_size = 128                # 논문에 언급 없음
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.merge_mode = merge_mode

        self.bi_gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                             num_layers=self.max_seq_len, batch_first=True, dropout=self.dropout_rate, bidirectional=True)

        self.bi_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                               num_layers=self.max_seq_len, batch_first=True, bidirectional=True)

        self.hidden = None

    def forward(self, x):
        if self.model == "gru":
            output, self.hidden = self.bi_gru(x)
        elif self.model == "lstm":
            output, self.hidden = self.bi_lstm(x)

        # # 파라미터 개수 확인 코드
        # cnt = 0
        # for param in model.parameters():
        #     param = torch.flatten(param)
        #     cnt += param.size(dim=0)
        # print("bigru weight counting: ", cnt)
        # bigru weight counting:  20763266, summary에 안뜨는건 GRU인지 LSTM인지 결정이 안되서 그런듯


        batch_size = x.size(dim=0)
        hidden_forward = self.hidden.view(2, self.max_seq_len, batch_size, self.hidden_size)[0]
        hidden_backward = self.hidden.view(2, self.max_seq_len, batch_size, self.hidden_size)[1]

        if self.merge_mode == "add":
            self.hidden = torch.add(hidden_forward, hidden_backward)
        elif self.merge_mode == "mean":
            self.hidden = torch.mean((hidden_forward, hidden_backward), dim=0)              # dimension 맞는지 재확인 필요

        return self.hidden


class Attention(nn.Module):         # feed-forward attention
    def __init__(self, d=128):
        super(Attention, self).__init__()
        self.d = d             # CNN: the number of filters = 128  /  Bi LSTM: dim of the contextual word embedding = 128
        # self.T = t             # length of generated feature map   /  Bi LSTM: the number of hidden states

        self.feedforward = nn.Linear(in_features=self.d, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, x):
        if x.size(dim=-1) == 128:                   # BiGRU
            x = x.permute(1, 0, 2)
        else:                                       # CNN
            x = torch.transpose(x, -1, -2)

        m = self.feedforward(x)
        m = self.tanh(m)
        alpha = self.softmax(m)
        r = torch.bmm(torch.transpose(x, -1, -2), alpha)

        return r


class Model(nn.Module):
    def __init__(self, max_seq_length, dropout_rate, embedding_size, vocab_size=None, merge_mode=None):
        super(Model, self).__init__()

        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.merge_mode = merge_mode

        self.cnn_attn = nn.Sequential(CNN(max_seq_len=max_seq_length,
                                          vocab_size=vocab_size, embedding_dim=embedding_size),
                                      Attention())

        self.bigru_attn = nn.Sequential(BiRNN(model="gru", max_seq_len=max_seq_length, dropout_rate=dropout_rate,
                                              merge_mode=merge_mode, embedding_dim=embedding_size),
                                        Attention())

        self.softmax = nn.Softmax()

    def forward(self, x):
        cnn_res = self.cnn_attn(x)
        bigru_res = self.bigru_attn(x)

        concated_res = torch.concat([cnn_res, bigru_res], dim=0)

        res = self.softmax(concated_res)        # dim=1 추가하기

        return res


# if __name__ == "__main__":
#     merge_mode = "concat"                       # concat, mean, add  --> 더 찾아보기
#     model = Model(max_seq_length=30, dropout_rate=1e-3, vocab_size=300, merge_mode=merge_mode, embedding_size=100)
#     # tmp = torch.rand(3, 30, 100)
#     # print(tmp.size())
#     # x = model(tmp)
#
#     from torchsummary import summary
#
#     # print("print(model): ", model)
#     print("summary(model, (64, 30, 100)): ", summary(model, (30, 100), device="cpu"))
#
#     from torchviz import make_dot
#     x = torch.zeros(1, 30, 100)
#     make_dot(model(x), params=dict(list(model.named_parameters())))


