import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pad_sequence
import math
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering
from tqdm import tqdm
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CNN1d(nn.Module):
    def __init__(self):
        super(CNN1d, self).__init__()
        self.cnn1d = nn.Conv1d()

    def forward(self, x):
        return

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
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        # self.bert_base = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        # self.bert_base =
        self.embedding_dim = embedding_dim
        # self.pos_encoder = PositionalEncoding()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8, batch_first=True)
        # self.layernorm = nn.LayerNorm(normalized_shape=[1,embedding_dim])
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)
        self.feedforward = nn.Linear(self.embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.encoder(x)
        cls_out = torch.mean(out, dim=-2)
        cls_out = self.feedforward(cls_out)
        cls_out = self.sigmoid(cls_out)
        return out, cls_out


class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()
        # self.bert = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        self.embedding_dim = embedding_dim
        # self.layernorm = nn.LayerNorm(normalized_shape=[embedding_dim])

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.embedding_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=6)

    def forward(self, tgt, memory):
        out = self.decoder(tgt, memory)

        return out

class AlzhBERT(nn.Module):
    def __init__(self, embedding_dim):
        super(AlzhBERT, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_sent_length = 7

        self.token_level_attn = nn.ModuleList([SelfAttention(self.embedding_dim, num_heads=8) for _ in range(10)]).requires_grad_(True)
        self.token_level_attn_single = SelfAttention(self.embedding_dim, num_heads=8).requires_grad_(True)
        self.sentence_level_attn = SelfAttention(self.embedding_dim, num_heads=8).requires_grad_(True)

        self.encoder = Encoder(embedding_dim=embedding_dim).requires_grad_(True)
        self.decoder = Decoder(embedding_dim=embedding_dim).requires_grad_(True)

    def forward(self, X_batch):
        i = 0

        enc_outs = []
        dec_outs = []
        for datastruct in tqdm(X_batch):
            j=0
            for section in datastruct.sections:
                # print(i, " + ", j)
                inv = section.inv.to(device)
                y_dec = section.next_uttr.to(device)
                par = section.par
                # print(par)
                try:
                    tmp = par.dim()
                except AttributeError:
                    # print(par)
                    # print("attr err")
                    j = j+1
                    continue

                # par = par.permute(1,0,2)                # (seq_len, sent_len, embed) => 한 번에 self attention
                # 여러개 self_attention
                # for p in par:
                result = self.token_level_attn_single(par.to(device))[0]
                res = torch.mean(result, dim=-2).unsqueeze(0)

                res_sent = self.sentence_level_attn(res.to(device))[0]
                context = torch.mean(res_sent, dim=-3)

                inv_input = torch.mean(inv, dim=-2)
                # x_enc = torch.concat((inv_input, context))
                # x_enc = x_enc.view([1, -1, self.embedding_dim])

                enc_out, cls_out = self.encoder(torch.cat([inv_input, context]).unsqueeze(0))
                # enc_out, cls_out = self.encoder(x_enc)
                # y_dec = torch.mean(y_dec, dim=-2).to(device)
                # enc_out = torch.mean(enc_out, dim=-2).unsqueeze(0).to(device)
                dec_out = self.decoder(y_dec, enc_out.to(device))

                cls_out = torch.mean(cls_out)
                enc_outs.append(cls_out)
                dec_outs.append(dec_out)

        return enc_outs, dec_outs
