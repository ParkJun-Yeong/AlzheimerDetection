import torch
import torch.nn as nn
from dataclasses import dataclass   # 구조체
from embedding import Embedding
from transformers import BertModel, BertConfig

import random
from torch.nn.utils.rnn import pad_sequence
import math
# from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering
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

def position_encoding():
    return

"""
Huggingface BertModel

- vocab_size = 30522
- hidden_size = 768
- num_hidden_layers = 12
- num_attention_heads = 12
- intermediate_size = 3072
- hidden_act = 'gelu'
- hidden_dropout_prob = 0.1
- attention_probs_dropout_prob = 0.1
- max_position_embeddings = 512
- type_vocab_size = 2
- initializer_range = 0.02
- layer_norm_eps = 1e-12
- pad_token_id = 0
- position_embedding_type = 'absolute'
- use_cache = True
- classifier_dropout = None
"""
class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.encoder = BertModel(self.config)
        self.sigmoid = nn.Sigmoid()
        self.feedforward = nn.Linear()

    def forward(self, x):
        input = position_encoding(x)
        out = self.encoder(input_ids=input, attention_mask, token_type_ids, position_ids)
        cls_out = torch.mean(out, dim=-2)
        cls_out = self.feedforward(cls_out)
        cls_out = self.sigmoid(cls_out)
        return out, cls_out


class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()
        # self.bert = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=6)

    def forward(self, tgt, memory):
        out = self.decoder(tgt, memory)

        return out

"""
- max_token_num: 한 문장 내 최대 토큰 개수
- max_seq_len: 입력 값으로 들어갈 문장의 최대 수 (첫번째 어탠션 네트워크 개수)
"""

class AlzhBERT(nn.Module):
    def __init__(self, max_token, max_seq, num_heads, embedding_dim=768):
        super(AlzhBERT, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_token = max_token
        self.max_seq = max_seq
        self.sec_div_flag = True               # Section을 나눌 것인지 아닌지 플래그
        # self.pred = pred        # True면 prdiction (cls_out, decoder_out, decoder_tgt) 리턴, False면 loss 리턴

        self.f = nn.ModuleList([SelfAttention(embedding_dim, num_heads=num_heads) for _ in range(max_seq)])     # token-level attention
        self.g = SelfAttention(embedding_dim, num_heads=num_heads)              # sentence-level attention

        self.encoder = Encoder()
        self.decoder = Decoder()


    def calculate_loss(self, pred, tgt):
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred.to(torch.float32), tgt.to(torch.float32))
        return loss

    """
    batch: DataStruct 리스트 (배치) or Dialogue 리스트
    Xs: DataStruct
    X: 한 DataStruct의 한 Section
    D: Dialogue
    
    labels: batch의 라벨들 (DataStruct 당 하나)
    """
    def forward(self, batch, labels):
        preds_enc = []
        preds_dec = []

        for D, y in (batch, labels):
            if self.sec_div_flag:
                for sections in D.sections:
                    inv = sections.inv
                    par = sections.par
                    next_uttr = sections.next_uttr

                    # 임베딩
                    x_inv = Embedding.bert_embedding(inv, mode_token=True)
                    x_par = Embedding.bert_embedding(par, mode_token=True)
                    y_enc = y
                    y_dec = Embedding.bert_embedding(next_uttr, mode_token=True)

                    # token-level positional embedding
                    outputs = []
                    for i in range(self.max_seq_len):
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

            elif not self.sec_div_flag:
                idx = [i for i in range(len(D.dialogue)) if D.who[i] == 'INV']          # INV 발화 인덱스 (idx[0] = 0)

                for i in range(len(idx)):
                    if i == 0:
                        continue                # i=0 이면 발화를 나눌 수 없기 때문에 다음 턴으로 넘어감
                    x = D.dialogue[i-1:i]
                    x_inv =
                    x_inv



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











