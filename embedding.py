import os
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
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


def bert_embedding():
    embedding_dataframe = pd.DataFrame(["embedded_sentence", 'label'])

    if not os.path.isfile(corpus_path):            # corpus.csv 존재 확인
        pre = Preprocess(corpus_exist=0)
    else:
        pre = Preprocess(corpus_exist=1)

    # pre-trained model의 weight을 로드.
    model = BertModel.from_pretrained('bert-base-uncased')

    # model을 evaluation mode에 두어서 feed-forward operation을 통과하게 함.
    model.eval()

    tmp_sent_dataframe = pd.DataFrame(["embedded_sentence"])
    for i, sent in tqdm(enumerate(pre.corpus["sentence"]), desc="Extracting Bert Embeddings..."):
        indexed_tokens, segments_ids = pre.bert_tokenize(sent)

        # Convert inputs to Pytorch tensors.
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)

        token_embeddings = torch.stack(encoded_layers, dim=0)
        print("token_embeddings.size(): ", token_embeddings.size())

        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        # Word Vectors (concatenate)
        token_vecs_cat = []
        for token in token_embeddings:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            token_vecs_cat.append(cat_vec)

        # print("Shape is: %d x %d" % (len(token_vecs_cat),
        #                              len(token_vecs_cat[0])))

        # 문장 벡터 만들기.
        token_vecs = encoded_layers[11][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)

        tmp_sent_dataframe = tmp_sent_dataframe.append(pd.Series(sentence_embedding), ignore_index=True)

    embedding_dataframe["label"] = pre.corpus["label"]
    embedding_dataframe["embedded_sentence"] = tmp_sent_dataframe["embedded_sentence"]

    return embedding_dataframe

if __name__ == "__main__":
    df = bert_embedding()
    df.to_csv("./dataset/tmp.csv", sep=",")