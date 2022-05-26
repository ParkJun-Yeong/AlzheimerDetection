import os
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from pytorch_pretrained_bert import BertTokenizer
# nltk.download('punkt')


class Preprocess:
    def __init__(self):
        self.path = "./dataset/corpus.csv"

        if not os.path.isfile(self.path):  # corpus.csv 존재 확인
            Preprocess.load_raw()

        self.corpus = pd.read_csv(self.path)
        # self.vocab = None

        self.corpus["sentence"] = self.corpus["sentence"].astype("string")

    @staticmethod
    # Read raw .txt data and Convert to dataframe
    def load_raw():
        columns = ['data_path', 'sentence', 'label']
        path = {"Dementia": "./dataset/dementia",
                "Control": "./dataset/control"}

        dementia_files = os.listdir(path["Dementia"])
        control_files = os.listdir(path["Control"])

        dataframe = pd.DataFrame(columns=columns)

        # Dementia: 1
        for file in tqdm(dementia_files, desc="Extracting dementia sequence."):
            file = os.path.join(path["Dementia"], file)
            document = pd.read_csv(file, header=None, sep='\n')

            for i in range(len(document)):
                sent = document.iloc[i, 0]
                dataframe = dataframe.append({"data_path": file, "sentence": sent,
                                  "label": 1}, ignore_index=True)

        # Control: 0
        for file in tqdm(control_files, desc="Extracting control sequence..."):
            file = os.path.join(path["Control"], file)
            document = pd.read_csv(file, header=None, sep='\n')

            for i in range(len(document)):
                sent = document.iloc[i, 0]
                dataframe = dataframe.append({"data_path": file, "sentence": sent,
                                  "label": 0}, ignore_index=True)

        # csv 저장
        csv_filename = "./dataset/corpus.csv"
        print("Save to \"", csv_filename, "\"")
        dataframe.to_csv(csv_filename, sep=',', na_rep="NaN", index_label="index")

    # 대문자를 소문자로 변환
    def lowercase(self):
        for i, sent in tqdm(enumerate(self.corpus["sentence"]), desc="processing lower casing..."):
            self.corpus["sentence"][i] = sent.lower()

    # bert 임베딩이 아닐 경우, corpus 문장 전체를 토큰화 해 vocab 생성
    def tokenize(self):
        # 입력 corpus에 대해서 NLTK를 이용해 문장 토큰화 (생략. .csv 변환 과정에서 이미 문장 토큰화 완료.)
        # sent_text = sent_tokenize((corpus_text))

        # 각 문장에 대해서 NLTK를 이용해 단어 토큰화
        vocab = []
        tokenized_sent = []
        for sent in tqdm(self.corpus["sentence"], desc="Tokenizing words and Making vocabulary..."):
            vocab.extend(word_tokenize(sent))
            tokenized_sent.append(word_tokenize(sent))

        vocab = set(vocab)

        return tokenized_sent, vocab

    @staticmethod
    # BERT Embedding 할 때 사용.
    def bert_tokenize(self, sent):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # for sent in tqdm(self.corpus["sentence"], desc="Bert Tokenizaiton"):
        marked_text = "[CLS]" + sent + "[SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        for tup in zip(tokenized_text, indexed_tokens):
            print('{:<12} {:>6,}'.format(tup[0], tup[1]))

            segments_ids = [1] * len(tokenized_text)

        return indexed_tokens, segments_ids

    def call(self):
        self.lowercase()


# if __name__ == "__main__":