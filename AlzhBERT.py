import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pad_sequence
import math
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering
from tqdm import tqdm
import pandas as pd


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(SelfAttention, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads,
                                                    batch_first=True)

    def forward(self, x):
        query = x
        key = x
        value = x
        attn_output = self.multihead_attn(query, key, value, need_weights=False)

        return attn_output

# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model=768, vocab_size=5000, dropout=0.1, batch_size=32):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(vocab_size, d_model)
#         position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, xs):
#         for x in xs:
#             x = x + self.pe[:, :x.size(1), :]
#             return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.bert_base = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        # self.bert_base =
        self.embedding_dim = 768
        # self.pos_encoder = PositionalEncoding()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)
        self.feedforward = nn.Linear(self.embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # sent_embed = [0]*self.embedding_dim + [1]*self.embedding_dim
        # input = torch.add(x, sent_embed)
        # input = torch


        # attention_masks = []
        # for seq in x:
        #     seq_mask = [float(i > 0) for i in seq]
        #     attention_masks.append(seq_mask)
        # attention_masks = torch.tensor(attention_masks)
        #
        # self.bert(input)
        # model.cuda()
        # x = self.pos_encoder(x)
        out = self.encoder(x)
        cls_out = torch.mean(out, dim=-2)
        cls_out = self.feedforward(cls_out)
        cls_out = self.sigmoid(cls_out)
        return out, cls_out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.bert = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        self.embedding_dim = 768
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.embedding_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=6)

    def forward(self, tgt, memory):
        out = self.decoder(tgt, memory)

        return out

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
            idx += 1

        inputs = []
        for i in range(len(idxs)-1):
            start = idxs[i]
            end = idxs[i + 1]
            input = {"who": dialogue["who"][start:end],
                     "sentence": dialogue["sentence"][start:end]}
            inputs.append(input)

        ret = []
        for cnt, input in enumerate(inputs):
            inv = input["sentence"][0].squeeze()        # all input: [seq_len, 768]
            par = input["sentence"][1:]
            outputs = []
            for i in range(10):
                try:
                    output = self.token_level_attn[i](par[i].unsqueeze(0))[0]
                except IndexError:
                    t = random.randint(0, par.shape[0] - 1)
                    output = self.token_level_attn[i](par[t].unsqueeze(0))[0]
                outputs.append(output)

            context = torch.concat(outputs, dim=-3)
            context = self.sentence_level_attn(context)[0]
            context = torch.mean(context, dim=-3).unsqueeze(0)
            context = torch.mean(context, dim=-2)

            inv_input = torch.mean(inv, dim=-2).unsqueeze(0)
            # decoder_tgt = torch.mean(inv_uttr[cnt+1], dim=-2).unsqueeze(0).unsqueeze(0)
            decoder_tgt = inv_uttr[cnt+1].unsqueeze(0)

            encoder_out, cls_out = self.encoder(torch.concat([inv_input, context]).unsqueeze(0))
            decoder_out = self.decoder(decoder_tgt, encoder_out)

            ret.append({"cls_out": cls_out, "decoder_out": decoder_out, "decoder_tgt": decoder_tgt})

        return ret











