import torch
import torch.nn as nn

# 1-dimensional CNN
class CNN(nn.Module):
    def __init__(self, max_seq_len, dropout_rate=0, vocab_size=1751, embedding_dim=100):
        super(CNN, self).__init__()
        self.max_seq_len = max_seq_len              # sentence length
        self.embedding_dim = embedding_dim          # GloVe, Word2Vec 기준으로 범위 잡기
        self.dropout_rate = dropout_rate

        # self.embbed = nn.Embedding(vocab_size, embedding_dim)
        # self.multi_size_conv = [nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(kernel_size, embedding_dim), stride=1) for kernel_size in range(1, 6)]     # first conv. layer
        # self.uni_size_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(3, embedding_dim), stride=1)
        # self.cnn = nn.ModuleList([self.multi_size_conv, self.uni_size_conv, self.uni_size_conv])

        self.relu = nn.ReLU()

        self.conv1 = nn.ModuleList([nn.Conv1d(in_channels=self.embedding_dim, out_channels=128,
                                                    kernel_size=kernel) for kernel in range(1, 6)])       # First CNN layers
        self.max1d = None

    def max_pooling(self, input, kernel, stride):
        self.max1d = nn.MaxPool1d(kernel, stride)
        ret = self.max1d(input)

        return ret

    def conv(self, input, in_channel, kernel, out_channel=128):
        layer = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel)
        ret = layer(input)

        return ret

    def forward(self, input):           # data loader 어떻게 사용하는지 추가하기

        input = input.permute(0, 2, 1)

        x = []
        for cnn in self.conv1:
            feature = cnn(input)          # Embedded Input
            feature = self.relu(feature)
            feature = self.max_pooling(feature, kernel=2, stride=2)
            x.append(feature)

        x = torch.cat(x, dim=2)
        x = self.conv(x, in_channel=x.size(dim=1), kernel=3)      # 논문 '그림 상'으로는 128이 in channel이므로 reshape 안함
        x = self.relu(x)
        x = self.max_pooling(x, kernel=2, stride=2)

        x = self.conv(x, in_channel=x.size(dim=1), kernel=3)
        x = self.relu(x)
        x = self.max_pooling(x, kernel=2, stride=2)

        return x

    def call(self):
        ret = self.forward()

        return ret


class BiRNN(nn.Module):
    def __init__(self, model, max_seq_len, embedding_dim=100):
        super(BiRNN, self).__init__()
        self.model = model
        self.embedding_dim = embedding_dim
        self.hidden_size = 100                # 논문에 언급 없음
        self.hidden_units = 128

        self.bi_gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                             num_layers=self.hidden_units, batch_first=True, bidirectional=True)

        self.bi_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                               num_layers=self.hidden_units, batch_first=True, bidirectional=True)

    def forward(self, x):
        if self.model == "gru":
            output, hidden = self.bi_gru(x)
        elif self.model == "lstm":
            output, hidden = self.bi_lstm(x)

        return output, hidden

    def call(self):
        ret = self.forward()

        return ret


class Attention(nn.Module):         # feed-forward attention
    def __init__(self, d, t):
        super(Attention, self).__init__()
        self.d = d             # CNN: the number of filters = 128  /  Bi LSTM: dim of the contextual word embedding = 128
        self.T = t             # length of generated feature map   /  Bi LSTM: the number of hidden states

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.feedforward = nn.Linear(in_features=self.d, out_features=1)

    def forward(self, x):
        x = torch.transpose(x, -1, -2)

        m = self.feedforward(x)
        m = self.tanh(m)
        alpha = self.softmax(m)
        r = torch.bmm(torch.transpose(x, -1, -2), alpha)

        return r

    def call(self):
        ret = self.fowrad()

        return ret

# Test run
if __name__ == "__main__":

    vocab_size = 1751   # 우선 논문과 동일하게 설정
    embedding_dim = 100    # 100-GloVe 기준

    embed = nn.Embedding(vocab_size, embedding_dim)