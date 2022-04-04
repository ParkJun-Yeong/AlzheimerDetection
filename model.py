import torch
from torch import nn
import torch.nn.functional as F
from utility import auto_rnn_bilstm, fit_seq_max_len, auto_rnn_bigru


"""
Bi-GRU
"""
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=100, dropout_rate=0.1, embed_weight=None):
        super(BiGRU, self).__init__()
        # self.n_layers = 1
        # self.hidden_size = hidden_size

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, _weight=embedding_weight)

        self.bi_gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.embedding_dim,
                            num_layers=1, batch_first=True, dropout=dropout_rate, bidrectional=True)

        # self.reverse_gru = torch.nn.GRU(input_size=self.embedding_dim, hidden_size=self.embedding_dim,
        #                                 num_layers=1, batch_first=True, dropout_rate=dropout_rate, bidirectional=False)
        #
        # self.reverse_gru.weight_ih_10 = self.bi_gru.weight_ih_10_reverse
        # self.reverse_gru.weight_hh_10 = self.bi_gru.weight_hh_10_reverse
        # self.reverse_gru.bias_ih_10 = self.bi_gru.weight_ih_10_reverse
        # self.reverse_gru.bias_hh_10 = self.bi_gru.weight_hh_10_reverse

    def forward(self, input_seq, input_length, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN modules
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_length)
        # Forward pass through GRU
        outputs, hidden = self.bi_gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:,:,:self.embedding_dim] + outputs[:,:,self.embedding_dim:]
        # Return output and final hidden state
        return outputs, hidden

"""
1D CNN
"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

    def forward(self):

"""
Attention
"""
class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1),
                                      encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


