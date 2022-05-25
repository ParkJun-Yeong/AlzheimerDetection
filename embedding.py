import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

"""
- Word2Vec
- Pre-trained Word2Vec
- Bert Embedding: 문맥에 따라 같은 의미의 단어도 다른 임베딩을 부여. (다른 단어 임베딩은 같은 단어 = 같은 임베딩)
"""

corpus_path = "./dataset/corpus.csv"
corpus = pd.read_csv(corpus_path)
corpus_len = len(corpus)

