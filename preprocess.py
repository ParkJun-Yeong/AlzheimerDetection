import os
import pandas as pd
import xml.etree.ElementTree as elemTree
from bs4 import BeautifulSoup as bf
import lxml
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
# from pytorch_pretrained_bert import BertTokenizer

nltk.download('punkt')


class Preprocess:
    def __init__(self):
        # self.path = "./dataset/corpus.csv"
        self.path = "./dataset/corpus.csv"

        # if not os.path.isfile(self.path):  # corpus.csv 존재 확인
        #     Preprocess.load_raw()

        if not os.path.isfile(self.path):  # corpus.csv 존재 확인
            Preprocess.xml_to_csv()

        self.corpus = pd.read_csv(self.path)
        # self.vocab = None

        self.corpus["sentence"] = self.corpus["sentence"].astype("string")

    """
        Read raw .txt data and Convert to dataframe
        .cha -> .cex -> .txt 변환 후 .cex -> .csv로 정리하기 위해 만든 메소드
        data_path, sentence, label 열을 가진 .csv가 리턴됨.
        """
    @staticmethod
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

    """
    ".cha" is raw file extension of the cookie theft corpus.
    In "add_tag()" method, read ".cha" files and add the tag [PAR] or [INV] to utterances following *PAR or *INV.
    Investigator는 환자의 메타데이터와 동일하게 저장함. (개입자의 메타데이터는 언어, 포지션 밖에 없기 때문)
    """
    @staticmethod
    def xml_to_csv():
        columns = ['file_num', 'file', 'uid', 'who', 'group', 'sex', 'age',
                   'sentence',  'language', 'education', 'label']
        path = {"Dementia": "./dataset/xml/dementia",
                "Control": "./dataset/xml/control"}

        dementia_files = os.listdir(path["Dementia"])
        control_files = os.listdir(path["Control"])

        df = pd.DataFrame(columns=columns)

        # Dementia: 1
        file_num = 0
        for file in tqdm(dementia_files, desc="Extracting dementia sequence."):
            file = os.path.join(path["Dementia"], file)

            with open(file, 'r', encoding='UTF8') as f:
                tree = bf(f, 'xml')
                part = tree.find_all('participant')[0]
                key = ['language', 'age', 'sex', 'group', 'education']
                is_key = [k in part.attrs for k in key]

                value = []
                for k, i in zip(key, is_key):
                    if not i:
                        value.append('NaN')
                    else:
                        value.append(part.attrs[k] if k != 'age' else int(part.attrs[k][1:3]))

                lan = value[0]
                age = value[1]
                sex = value[2]
                group = value[3]
                edu = value[4]

                # if 'age' in part.attrs:
                #     age = int(part.attrs['age'][1:3])
                # else:
                #     age = 'NaN'
                # sex = part.attrs['sex']
                # group = part.attrs['group']
                # if 'education' in part.attrs:
                #     age = int(part.attrs['age'][1:3])
                # else:
                #     age = 'NaN'
                # edu = part.attrs['education']

                uttrs = tree.find_all('u')

                for u in uttrs:
                    uid = u.attrs['uID']
                    who = u.attrs['who']

                    words = u.find_all('w')
                    sentence = [str(w.contents[0]) for w in words]
                    sentence = " ".join(sentence)

                    df = df.append({'file': file, 'uid': uid, 'who': who,
                                    'group': group, 'sex': sex, 'age': age,
                                    'sentence': sentence, 'language': lan,
                                    'education': edu, 'label': 1, 'file_num': file_num}, ignore_index=True)
            file_num += 1

        # Control: 0
        for file in tqdm(control_files, desc="Extracting control sequence."):
            file = os.path.join(path["Control"], file)

            with open(file, 'r', encoding='UTF8') as f:
                tree = bf(f, 'xml')
                part = tree.find_all('participant')[0]
                key = ['language', 'age', 'sex', 'group', 'education']
                is_key = [k in part.attrs for k in key]

                value = []
                for k, i in zip(key, is_key):
                    if not i:
                        value.append('NaN')
                    else:
                        value.append(part.attrs[k] if k != 'age' else int(part.attrs[k][1:3]))

                lan = value[0]
                age = value[1]
                sex = value[2]
                group = value[3]
                edu = value[4]

                uttrs = tree.find_all('u')

                for u in uttrs:
                    uid = u.attrs['uID']
                    who = u.attrs['who']

                    words = u.find_all('w')
                    sentence = [str(w.contents[0]) for w in words]
                    sentence = " ".join(sentence)

                    df = df.append({'file': file, 'uid': uid, 'who': who,
                                    'group': group, 'sex': sex, 'age': age,
                                    'sentence': sentence, 'language': lan,
                                    'education': edu, 'label': 0, 'file_num': file_num}, ignore_index=True)

            file_num += 1

        # csv 저장
        csv_filename = "./dataset/xml/corpus.csv"
        print("Save to \"", csv_filename, "\"")
        df.to_csv(csv_filename, sep=',', na_rep="NaN", index_label="index")

    """
    PAR로 시작하는 대화 118개
    모든 대화는 INV의 "그림을 묘사해주세요" 등과 같은 말로 시작해야 함.
    해당 대화의 첫 마디를 "just tell me everything that you see happening in that picture"로 추가 (file:0 의 INV 시작발화)
    """
    @staticmethod
    def fill_inv(corp):
        inv_uttr = "just tell me everything that you see happening in that picture"
        corpus = corp

        for i in range(552):
            file = corpus.loc[corpus['file_num'] == i, :]
            if file.iloc[0]['who'] == 'PAR':
                idx = file.index[0]
                row = corpus.iloc[idx]

                inv_df = pd.DataFrame([{'index': -1, 'file_num': row['file_num'], 'file': row['file'],
                                        'uid': 'other', 'who': "INV", 'group': row['group'], 'sex': row['sex'],
                                        'age': row['age'], 'sentence': inv_uttr, 'language': row['language'],
                                        'education': row['education'], 'label': row['label']}])

                corpus = pd.concat([corpus.iloc[:idx], inv_df, corpus.iloc[idx:]], ignore_index=True)

        corpus.reset_index(inplace=True)

        return corpus

        # # Test code
        # file = os.path.join(in_path, files[0])
        # with open(file, 'r') as f:
        #     tree = bf(f, 'xml')
        #
        #     utter = tree.find_all('w')



        # for file in files[:2]:
        #     file = os.path.join(in_path, file)
        #     with open(file, 'r') as f:
        #         tree = bf(f, 'lxml')
        #         utter = tree.findall('w')
            # tree = elemTree.parse(file)
            # root = tree.getroot()
            #
            # utterance = tree.findall("CHAT")


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

    # BERT Embedding 할 때 사용.
    """
    order: 몇 번째 문장인지
    """
    @staticmethod
    def bert_tokenize(sent, order, verbose=False):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # for sent in tqdm(self.corpus["sentence"], desc="Bert Tokenizaiton"):
        if order == 0:
            marked_text = "[CLS]" + str(sent) + "[SEP]"
        else:
            marked_text = str(sent) + "[SEP]"

        tokenized_text = tokenizer.tokenize(marked_text, return_tensors='pt')

        return tokenized_text

    def call(self):
        self.lowercase()


if __name__ == "__main__":
    Preprocess.xml_to_csv()
