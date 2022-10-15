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
    def __init__(self, embedding_dim, num_kernels, kernel_size, stride):
        super(CNN1d, self).__init__()
        self.cnn1d = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim,
                                              kernel_size=kernel_size, stride=stride).to(device) for _ in range(num_kernels)])
        self.max1d = nn.MaxPool1d(kernel_size=2, stride=2).to(device)
        self.relu = nn.ReLU()
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = x.permute(0,2,1)

        feature_map = []
        for cnn in self.cnn1d:
            feature = cnn(x)
            feature = self.relu(feature)
            feature = self.max1d(feature).to(device)
            feature_map.append(feature)

        feature_map = torch.stack(feature_map)

        return feature_map

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
    def __init__(self, mode, embedding_dim=100):
        super(AlzhBERT, self).__init__()
        super(AlzhBERT, self).__init__()

        self.CONFIG_ = {"FIRST_MODULE_": mode,  # single of multi or cnn
                        "EMBEDDING_": "bert"}

        if self.CONFIG_["EMBEDDING_"] == "bert":
            self.embedding_dim = 768
        else:
            self.embedding_dim = embedding_dim

        self.max_sent_length = 10

        if self.CONFIG_["FIRST_MODULE_"] == "multi":
            self.token_level_attn = nn.ModuleList([SelfAttention(self.embedding_dim, num_heads=8) for _ in range(self.max_sent_length)]).requires_grad_(True)
        elif self.CONFIG_["FIRST_MODULE_"] == "single":
            self.token_level_attn = SelfAttention(self.embedding_dim, num_heads=8).requires_grad_(True)
        elif self.CONFIG_["FIRST_MODULE_"] == "cnn":
            self.token_level_cnns = nn.ModuleList([CNN1d(embedding_dim=embedding_dim, num_kernels=32, kernel_size=size, stride=1) for size in range(1,6)])

        self.sentence_level_attn = SelfAttention(self.embedding_dim, num_heads=8).requires_grad_(True)

        self.encoder = Encoder(embedding_dim=embedding_dim).requires_grad_(True)
        self.decoder = Decoder(embedding_dim=embedding_dim).requires_grad_(True)

    def forward(self, X_batch):
        i = 0

        enc_outs = []
        dec_outs = []
        for datastruct in X_batch:
            j=0
            for section in datastruct.sections:
                inv = section.inv.to(device)
                y_dec = section.next_uttr.to(device)
                par = section.par
                try:
                    tmp = par.dim()
                except AttributeError:
                    j = j+1
                    continue

                enc_out = None
                dec_out = None
                cls_out = None
                inv_input = None
                context = None

                if self.CONFIG_["FIRST_MODULE_"] == "single":
                    result = self.token_level_attn(par.to(device))[0]
                    res = torch.mean(result, dim=-2).unsqueeze(0)

                    res_sent = self.sentence_level_attn(res.to(device))[0]
                    context = torch.mean(res_sent, dim=-3)


                if self.CONFIG_["FIRST_MODULE_"] == "multi":
                    outputs = []
                    for i in range(self.max_sent_length):
                        try:
                            output = self.token_level_attn[i](par[i].to(device))[0]
                        except IndexError:
                            pad_tensor = torch.zeros_like(par[0]).to(device)
                            output = self.token_level_attn[i](pad_tensor)[0].to(device)
                        outputs.append(output)
                    context = torch.stack(outputs).to(device)

                if self.CONFIG_["FIRST_MODULE_"] == "cnn":
                    outputs = []
                    for conv in self.token_level_cnns:
                        output = conv(par.to(device))
                        output = torch.mean(output, dim=0)
                        output = output.permute(2,0,1)
                        outputs.append(output)
                    outputs = pad_sequence(outputs, batch_first=False)
                    outputs = torch.mean(outputs, dim=0)
                    context = outputs.transpose(0,1)

                    context = self.sentence_level_attn(context)[0]
                    context = torch.mean(context, dim=-3).unsqueeze(0)
                    context = torch.mean(context, dim=-2)

                inv_input = torch.mean(inv, dim=-2)
                enc_out, cls_out = self.encoder(torch.cat([inv_input, context]).unsqueeze(0))
                dec_out = self.decoder(y_dec, enc_out.to(device))
                cls_out = torch.mean(cls_out)

                enc_outs.append(cls_out)
                dec_outs.append(dec_out)

        return enc_outs, dec_outs
