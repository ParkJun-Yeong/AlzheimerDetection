import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pad_sequence
import math
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering
from tqdm import tqdm
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
    def __init__(self, embedding_dim=768, pred=False):
        super(AlzhBERT, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_sent_length = 7
        self.pred = pred        # True면 prdiction (cls_out, decoder_out, decoder_tgt) 리턴, False면 loss 리턴

        self.token_level_attn = nn.ModuleList([SelfAttention(self.embedding_dim, num_heads=8) for _ in range(10)])
        self.sentence_level_attn = SelfAttention(self.embedding_dim, num_heads=8)

        self.encoder = Encoder()
        self.decoder = Decoder()

    def calculate_loss(self, pred, tgt):
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred.to(torch.float32), tgt.to(torch.float32))
        return loss

    # x is dictionary (dialogue)
    def forward(self, batch, y_batch, valid=False):

        if self.pred == False:
            loss = {'enc': 0, 'dec': 0}
        if valid:
            accuracy = 0.0

        sample_num = 0
        for dialogue, cls_label in zip(batch, y_batch):
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

                if len(par) == 0: continue

                for i in range(10):
                    try:
                        output = self.token_level_attn[i](par[i].unsqueeze(0).to(device))[0]
                    except IndexError:
                        t = random.randint(0, par.shape[0] - 1)
                        output = self.token_level_attn[i](par[t].unsqueeze(0).to(device))[0]
                    outputs.append(output)

                context = torch.concat(outputs, dim=-3)
                context = self.sentence_level_attn(context.to(device))[0]
                context = torch.mean(context, dim=-3).unsqueeze(0)
                context = torch.mean(context, dim=-2)

                inv_input = torch.mean(inv, dim=-2).unsqueeze(0).to(device)
                # decoder_tgt = torch.mean(inv_uttr[cnt+1], dim=-2).unsqueeze(0).unsqueeze(0)
                decoder_tgt = inv_uttr[cnt+1].unsqueeze(0).to(device)

                encoder_out, cls_out = self.encoder(torch.concat([inv_input, context]).unsqueeze(0))
                decoder_out = self.decoder(decoder_tgt, encoder_out.to(device))

                cls_out = cls_out.unsqueeze(0)
                loss['enc'] += self.calculate_loss(cls_out, cls_label)
                loss['dec'] += self.calculate_loss(decoder_out, decoder_tgt)
                sample_num += 1

                cls_pred = 1 if cls_out >= 0.5 else 0

                if valid:
                    if cls_pred == cls_label:
                        accuracy += 1

        if valid:
            accuracy = accuracy/float(sample_num)
            return loss['enc'], loss['dec'], accuracy, sample_num
        else:
            return loss['enc'], loss['dec'], sample_num











