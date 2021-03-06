import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
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

class Embedding:
    def __init__(self):
        pass

    """
    bert_embedding
    - (batch_sentences): None(corpus 데이터 모두 임베딩 후 리턴), NotNone(들어온 sentence 배치 임베딩 후 리턴)
    - (sent_embed): True(문장 단위 임베딩, 1x768 리턴), False(토큰 단위 임베딩, seqlen x768 리턴)
    """
    @staticmethod
    def bert_embedding(batch_sentences=None, sent_embed=False):

        # corpus sentence 전체 임베딩 -> .csv로 저장
        if batch_sentences is None:
            columns = ["embedded", "label"]
            embedding_dataframe = pd.DataFrame(columns=columns)

            pre = Preprocess()

            # pre-trained model의 weight을 로드.
            model = BertModel.from_pretrained('bert-base-uncased')

            # model을 evaluation mode에 두어서 feed-forward operation을 통과하게 함.
            model.eval()

            # tmp_sent_dataframe = pd.DataFrame(columns=[columns[0]])
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

                tmp_sent_dataframe = tmp_sent_dataframe.append({"embedded": [pd.Series(sentence_embedding)]}, ignore_index=True)

            embedding_dataframe["label"] = pre.corpus["label"]
            embedding_dataframe["embedded"] = tmp_sent_dataframe["embedded"]

            return embedding_dataframe

        else:                               # 배치 문장씩 임베딩 한 후 결과 반환
            model = BertModel.from_pretrained('bert-base-uncased')
            model.eval()

            batch_embedding = []
            for sentence in tqdm(batch_sentences, desc="Tokenizing and Embedding..."):
                indexed_tokens, segments_ids = Preprocess.bert_tokenize(sentence)

                # Convert inputs to Pytorch tensors.
                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segments_ids])

                with torch.no_grad():
                    encoded_layers, _ = model(tokens_tensor, segments_tensors)

                token_embeddings = torch.stack(encoded_layers, dim=0)
                # print("token_embeddings.size(): ", token_embeddings.size())

                token_embeddings = torch.squeeze(token_embeddings, dim=1)
                token_embeddings = token_embeddings.permute(1, 0, 2)

                # Word Vectors (concatenate)
                token_vecs_cat = []
                for token in token_embeddings:
                    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
                    token_vecs_cat.append(cat_vec)

                # print("Shape is: %d x %d" % (len(token_vecs_cat),
                #                              len(token_vecs_cat[0])))

                # 문장 벡터 만들기.
                token_vecs = encoded_layers[11][0]                  # (seq_len, 768)
                sentence_embedding = torch.mean(token_vecs, dim=0)
                sentence_embedding = torch.unsqueeze(input=sentence_embedding, dim=0)           # (1, 768)

                if sent_embed:
                    batch_embedding.append(sentence_embedding)
                else:
                    batch_embedding.append(token_vecs)

                # if batch_embedding is None:
                #     batch_embedding = sentence_embedding
                # else:
                #     batch_embedding = torch.concat((batch_embedding, sentence_embedding), dim=0)

            batch_embedding = pad_sequence(batch_embedding).permute(1, 0, 2)
            print("batch_embedding size: ", batch_embedding.size())

            return batch_embedding
#
#
# if __name__ == "__main__":
#     sentence 자동 처리 (create embedded.csv)
#     df = bert_embedding()
#     df.to_csv("./dataset/embedded.csv", sep=",")
#
#     한 문장씩 처리하는 bert embedding layer 만들기
