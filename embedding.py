import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from preprocess import Preprocess
from tqdm import tqdm

"""
- Word2Vec (https://wikidocs.net/50739)
- Bert Embedding: 문맥에 따라 같은 의미의 단어도 다른 임베딩을 부여. (다른 단어 임베딩은 같은 단어 = 같은 임베딩) (https://codlingual.tistory.com/98)
- Glove (https://wikidocs.net/22885)
"""

corpus_path = "./dataset/corpus.csv"
# corpus = pd.read_csv(corpus_path)
# corpus_len = len(corpus)

class Embedding:
    def __init__(self):
        pass

    """
    bert_embedding
    - (batch_sentences): None(corpus 데이터 모두 임베딩 후 리턴), NotNone(들어온 sentence 배치 임베딩 후 리턴)
    - (sent_embed): True(문장 단위 임베딩, 1x768 리턴), False(토큰 단위 임베딩, seqlen x768 리턴)
    
    - sequences: (N,num_tokens)
    - mode_token: 토큰 단위 임베딩, seqlen x 768 리턴
    - mode_sent: 문장 단위 임베딩, 1 x 768 리턴
    """
    @staticmethod
    def bert_embedding(sequences, mode_token=False, mode_sent=False):
        if (not mode_token) & (not mode_sent):
            print("Error: select mode")
            return -1

        if mode_token & mode_sent:
            print("Error: select only one mode")
            return -1

        try:
            num_batch = len(sequences)          # 리스트라면
            print("multiple sequence")
        except:
            num_batch = 1
            print("single sequence")

        model = BertModel.from_pretrained('bert-base-uncased')
        outputs = []

        for i in range(num_batch):
            indexed_tokens = Preprocess.bert_tokenize(sequences[i], order=i)

            with torch.no_grad():
                output = model(**indexed_tokens)[0].squeeze(0)

            if mode_token:
                outputs.append(output)         # (batch, seq_length, embedidng)
            elif mode_sent:
                output_mean = torch.mean(output, dim=1)     # (batch, embedding)
                outputs.append(output_mean)

        word_embedding = pad_sequence(outputs).permute(1, 0, 2)

        return word_embedding
#
#
# if __name__ == "__main__":
#     sentence 자동 처리 (create embedded.csv)
#     df = bert_embedding()
#     df.to_csv("./dataset/embedded.csv", sep=",")
#
#     한 문장씩 처리하는 bert embedding layer 만들기
