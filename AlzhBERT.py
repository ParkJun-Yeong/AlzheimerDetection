import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm
import pandas as pd


class SelfAttention(nn):
    def __init__(self, embedding_dim, num_heads):
        super(SelfAttention, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

    def forward(self, x):
        query = x
        key = x
        value = x
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)

        return attn_output, attn_output_weights

class Encoder(nn):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x):

class Decoder(nn):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):


class AlzhBERT(nn):
    def __init__(self, embedding_dim):
        super(AlzhBERT, self).__init__()
        self.embedding_dim = embedding_dim

        self.token_level_attn = None
        self.sentence_level_attn = None

        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, x):

