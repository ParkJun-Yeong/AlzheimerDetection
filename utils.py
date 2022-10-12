import torch
from embedding import Embedding
from torch.nn.utils.rnn import pad_sequence


"""
data: X_folds["train"/"valid"][i] => tuple 58
[(DataStruct tuple), (DataStruct tuple), ... ]

DataStruct 하나에 3개의 tensor로 만듦

Output: 3개의 tensor
"""
def batch_to_tensor(batch):
    length = len(batch)     # 58



    inv_tensor = []
    par_tensor = []
    next_tensor = []

    for datastructs in batch:
        invs_datastruct = []
        pars_datastruct = []
        nexts_datastruct = []

        for section in datastructs.sections:
            # inv_single_section
            par_single_section = []
            # next_single_section = []

            inv_single_section = Embedding.bert_embedding(sequences=section.inv, mode_token=True).squeeze(0)
            par_single_section.append([Embedding.bert_embedding(sequences=p, mode_token=True).squeeze(0) for p in section.par])
            par_single_section = par_single_section[0]

            if len(par_single_section) > 1:
                par_single_section = pad_sequence(par_single_section).permute(1,0,2)
            else:
                par_single_section = par_single_section[0].unsqueeze(0)

            next_single_section = Embedding.bert_embedding(sequences=section.next_uttr, mode_token=True).squeeze(0)

            invs_datastruct.append(inv_single_section)
            pars_datastruct.append(par_single_section)
            nexts_datastruct.append(next_single_section)

        inv_tensor.append(invs_datastruct)
        par_tensor.append(pars_datastruct)
        next_tensor.append(nexts_datastruct)

    return inv_tensor, par_tensor, next_tensor


def get_y_enc_list(datastruct_list, y):
    y_enc = []

    for i, datastruct in enumerate(datastruct_list):
        for section in datastruct.sections:
            if len(section.par) == 0:
                continue
            y_enc.append(y[i])

    return y_enc

"""
DataStruct list -> next_uttr 추출, 단 par이 없는 경우는 제외 -> y_dec 생성
"""
def get_y_dec_list(datastruct_list):
    y_dec = []

    for datastruct in datastruct_list:
        for section in datastruct.sections:
            if len(section.par) == 0:
                continue

            y_dec.append(section.next_uttr)

    return y_dec
