import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
# from transformers import BertModel
from pytorch_pretrained_bert import BertModel
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

        if isinstance(sequences, list):
            num_batch = len(sequences)          # 리스트라면
            print("multiple sequence")
        else:
            num_batch = 1
            sequences = [sequences]
            print("single sequence")

        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        outputs = []

        for i in range(num_batch):
            indexed_tokens, segments_ids = Preprocess.bert_tokenize(sequences[i], order=i)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensor = torch.tensor([segments_ids])

            with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, segments_tensor)

            token_embeddings = torch.stack(encoded_layers, dim=0).squeeze(1)
            # token_embeddings = token_embeddings.permute(1,0,2)

            # token_vecs_cat = []
            # for token in token_embeddings:
                # cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            # token_vecs_cat.append(token_embeddings)

            token_embeddings = encoded_layers[11]                # Last Hidden: (batch, seq_len, 768)
            # token_embeddings = torch.mean(token_embeddings, dim=0)              # Mean: (batch, seq_len, 768)


            if mode_token:
                if num_batch > 1:           # multiple sequences
                    outputs.append(token_embeddings)      # (batch, seq_length, embedidng)
                else:
                    outputs = token_embeddings            # (1, seq_length, embedding)
            elif mode_sent:
                sentence_embedding = torch.mean(token_embeddings, dim=-2)     # (batch, embedding)
                # sentence_embedding = sentence_embedding.unsqueeze()
                outputs.append(sentence_embedding)

        outputs = pad_sequence(outputs).permute(1, 0, 2)
        print("result embedding size: ", outputs.size())

        return outputs


    """
    sequences: 한 문장 혹은 여러 문장을 처리
    corpus: 전체 말뭉치 처리
    tokenizing은 dataset.py에서 처리
    입력은 tokenized data만 받음
    """
    @staticmethod
    def glove_embedding(sequences):
        glove = GloVe(name="840B", dim=300)
        pre = Preprocess()
        tokenized_sequences, vocab = pre.tokenize(sentences=sequences)

        embeddings = []
        for seq in tqdm(tokenized_sequences, desc="Tokenizing and Embedding...."):
            embedding = glove.get_vecs_by_tokens(seq, lower_case_backup=True)
            embeddings.append(embedding)

        embeddings = pad_sequence(embeddings).permute(1,0,2)
        print("COMPLETE:", embeddings.size())

        return embeddings




import pandas as pd
#
#
if __name__ == "__main__":
#     sentence 자동 처리 (create embedded.csv)
#     df = bert_embedding()
#     df.to_csv("./dataset/embedded.csv", sep=",")
#
#     한 문장씩 처리하는 bert embedding layer 만들기
    corpus = pd.read_csv("./dataset/corpus.csv")
    texts = corpus.loc[:, "sentence"].values.tolist()
    Embedding.glove_embedding(texts)
