import torch
import torch.nn as nn
from dataclasses import dataclass   # 구조체

import random
from torch.nn.utils.rnn import pad_sequence
import math
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering
from tqdm import tqdm
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@dataclass
class ResultStruct:
    pred_enc = None     # int
    pred_dec = None     # tensor


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(SelfAttention, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads,
                                                    batch_first=True)

    def forward(self, x):
        query = x
        key = x
        value = x
        attn_output, attn_output_weights = self.multihead_attn(query, key, value, need_weights=True)

        return attn_output, attn_output_weights


class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()

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

    def forward(self, tgt, memory):
        out = self.decoder(tgt, memory)

        return out


class AlzhBERT(nn.Module):
    def __init__(self, max_token_num, max_seq_len, num_heads, embedding_dim=768):
        super(AlzhBERT, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_token_len = max_token_num
        self.max_seq_len = max_seq_len
        # self.pred = pred        # True면 prdiction (cls_out, decoder_out, decoder_tgt) 리턴, False면 loss 리턴

        self.token_level_attn = nn.ModuleList([SelfAttention(embedding_dim, num_heads=num_heads) for _ in range(max_token_num)])
        self.sentence_level_attn = SelfAttention(embedding_dim, num_heads=num_heads)

        self.encoder = Encoder()
        self.decoder = Decoder()

    def calculate_loss(self, pred, tgt):
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred.to(torch.float32), tgt.to(torch.float32))
        return loss

    """
    batch: DataStruct 리스트 (배치)
    Xs: DataStruct 하나
    X: 한 DataStruct의 한 Section
    
    labels: batch의 라벨들 (DataStruct 당 하나)
    """
    def forward(self, batch, labels):
        preds_enc = []
        preds_dec = []

        for Xs, y in (batch, labels):
            for X in Xs.sections:
                inv = X.inv
                par = X.par
                next_uttr = X.next_uttr

                # 임베딩
                x_inv = embedding(inv)
                x_par = embedding(par)
                y_enc = y
                y_dec = embedding(next_uttr)

                # token-level positional embedding



                outputs = []
                for i in range(self.max_token_len):
                    attn_output, attn_weights = self.token_level_attn[i](par[i])
                    outputs.append(attn_output)

                context = torch.concat(outputs, dim=-3)

                # sentence-level positional embedding

                context = self.sentence_level_attn(context.to(device))[0]
                context = torch.mean(context, dim=-3).unsqueeze(0)
                context = torch.mean(context, dim=-2)

                # sentence embedding 추가

                out_enc, pred_enc = self.encoder(x_inv, context)
                pred_dec = self.decoder(out_enc)

                preds_enc.append(pred_enc)
                preds_dec.append(pred_dec)

        return preds_enc, preds_dec




            # ret = []
            # inv = input["sentence"][0].squeeze()        # all input: [seq_len, 768]
            #     par = input["sentence"][1:]
            #     outputs = []
            #
            #     if len(par) == 0: continue
            #
            #     for i in range(10):
            #         try:
            #             output = self.token_level_attn[i](par[i].unsqueeze(0).to(device))[0]
            #         except IndexError:
            #             t = random.randint(0, par.shape[0] - 1)
            #             output = self.token_level_attn[i](par[t].unsqueeze(0).to(device))[0]
            #         outputs.append(output)


                #
                # inv_input = torch.mean(inv, dim=-2).unsqueeze(0).to(device)
                # # decoder_tgt = torch.mean(inv_uttr[cnt+1], dim=-2).unsqueeze(0).unsqueeze(0)
                # decoder_tgt = inv_uttr[cnt+1].unsqueeze(0).to(device)
                #
                # encoder_out, cls_out = self.encoder(torch.concat([inv_input, context]).unsqueeze(0))
                # decoder_out = self.decoder(decoder_tgt, encoder_out.to(device))

        #         cls_out = cls_out.unsqueeze(0)
        #         loss['enc'] += self.calculate_loss(cls_out, cls_label)
        #         loss['dec'] += self.calculate_loss(decoder_out, decoder_tgt)
        #         sample_num += 1
        #
        #         cls_pred = 1 if cls_out >= 0.5 else 0
        #
        #         if valid:
        #             if cls_pred == cls_label:
        #                 accuracy += 1
        #
        # if valid:
        #     accuracy = accuracy/float(sample_num)
        #     return loss['enc'], loss['dec'], accuracy, sample_num
        # else:
        #     return loss['enc'], loss['dec'], sample_num











