import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm
import pandas as pd


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(SelfAttention, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

    def forward(self, x):
        query = x
        key = x
        value = x
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)

        return attn_output, attn_output_weights

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x):

class Decoder(nn):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):


class AlzhBERT(nn.Module):
    def __init__(self, embedding_dim=768):
        super(AlzhBERT, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_sent_length = 7

        self.token_level_attn = nn.ModuleList([SelfAttention(self.embedding_dim, num_heads=8) for _ in range(10)])
        self.sentence_level_attn = SelfAttention(self.embedding_dim, num_heads=8)

        self.encoder = Encoder()
        self.decoder = Decoder()


    # x is dictionary (dialogue)
    def forward(self, dialogue):

        inv_uttr = []
        idxs = []
        idx = 0
        for w in dialogue['who']:
            if w == "INV":
                idxs.append(idx)
                inv_uttr.append(dialogue["sentence"][idx])
            else:
                idx += 1

        inputs = []
        for i in range(len(idxs)-1):
            start = idxs[i]
            end = idxs[i + 1]
            input = {"who": dialogue["who"][start:end],
                     "sentence": dialogue["sentence"][start:end]}
            inputs.append(input)

        for input in inputs:
            inv = input["sentence"][0].squeeze()        # all input: [seq_len, 768]
            par = input["sentence"][1:]
            outputs = []
            for i in range(10):
                try:
                    output = self.token_level_attn[i](par[i])
                except IndexError:
                    output, _ = self.token_level_attn[i](par[int(i/2)])
                outputs.append(output)

            torch.concat(outputs)




