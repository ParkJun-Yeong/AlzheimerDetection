# Convert 'chatbot_demo.ipynb' to a single file.
# Ref.: https://doheon.github.io/%EC%BD%94%EB%93%9C%EA%B5%AC%ED%98%84/nlp/ci-chatbot-post/

import torch
import torch.nn as nn
import pandas as pd
import re
import urllib.request
import sentencepiece as spm
import numpy as np
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pd.set_option('mode.chained_assignment', None)  # SettingWithCopyWarning 경고 무시

"""
Preprocess part
"""
class Preprocess:
    def __init__(self):
        # urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
        # self.train_data = pd.read_csv('ChatBotData.csv')
        self.train_data = pd.read_csv("KETI_대화데이터_응급상황.txt", sep='\t', names=["index", "Q"])
        self.train_data['A'] = self.train_data['Q'].iloc[1:]
        for i in range(len(self.train_data)-1):
            self.train_data['A'].iloc[i] = self.train_data['A'].iloc[i+1]
        self.train_data = self.train_data.iloc[:-1]
        self.train_data.to_csv("./KETI_대화데이터_응급상황_QA.csv", sep=',', index=False)

    def remove_punctuation(self):
        print("Train data samples: \n", self.train_data.head())
        print("챗봇 샘플 개수: ", len(self.train_data), '\n')
        print("결측치 수: \n", self.train_data.isnull().sum())

        # 구두점 띄어쓰기
        questions = []
        for sentence in self.train_data['Q']:
            sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
            sentence = sentence.strip()
            questions.append(sentence)

        answers = []
        for sentence in self.train_data['A']:
            sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
            sentence = sentence.strip()
            answers.append(sentence)

        return questions, answers

    # Bert pretrained tokenize를 사용하지 않고 직접 토크나이징하는 방법.
    def sent_piece(self, corpus):
        prefix = "chatbot"
        vocab_size = 8000
        spm.SentencePieceTrainer.train(
            f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +
            " --model_type=bpe" +
            " --max_sentence_length=999999" +
            " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
            " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
            " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
            " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
            " --user_defined_symbols=[SEP],[CLS],[MASK]")   # 사용자 정의 토큰

        vocab_file = "chatbot.model"
        vocab = spm.SentencePieceProcessor()
        vocab.load(vocab_file)
        line = "안녕하세요 만나서 반갑습니다"          # Example
        pieces = vocab.encode_as_pieces(line)       # token = piece
        ids = vocab.encode_as_ids(line)             # piece(token)들의 단어 id 목록

        print("[Result Example]")
        print("Query: ", line)
        print("Piece(token): ", pieces)
        print("Id: ", ids)

        return vocab

    # 학습된 vocab을 이용해서 주어진 문장을 정수로 인코딩
    # 토큰화 / 정수인코딩 / 시작토큰과 종료토큰 추가 / 패딩
    def tokenize_and_filter(self, inputs, outputs, MAX_LENGTH, vocab):
        START_TOKEN = [2]
        END_TOKEN = [3]

        tokenized_inputs, tokenized_outputs = [], []

        for (sentence1, sentence2) in zip(inputs, outputs):
        # encdoe(토큰화 + 정수인코딩), 시작토큰과 종료토큰 추가
            zeros1 = np.zeros(MAX_LENGTH, dtype=int)
            zeros2 = np.zeros(MAX_LENGTH, dtype=int)
            sentence1 = START_TOKEN + vocab.encode_as_ids(sentence1) + END_TOKEN
            zeros1[:len(sentence1)] = sentence1[:MAX_LENGTH]

            sentence2 = START_TOKEN + vocab.encode_as_ids(sentence2) + END_TOKEN
            zeros2[:len(sentence2)] = sentence2[:MAX_LENGTH]

            tokenized_inputs.append(zeros1)
            tokenized_outputs.append(zeros2)

        return tokenized_inputs, tokenized_outputs


"""
Dataset part
"""
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, questions, answers):
        questions = np.array(questions)
        answers = np.array(answers)
        self.inputs = questions
        self.dec_inputs = answers[:, :-1]
        self.outputs = answers[:, 1:]
        self.length = len(questions)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.dec_inputs[idx], self.outputs[idx])

    def __len__(self):
        return self.length


"""
Model part
"""
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

class ChatbotModel(nn.Module):
    # ntoken: vocab의 size
    # embed: embedding size
    # nhead: head의 개수
    # hidden: feedforward 차원
    # nlayers: layer 개수
    def __init__(self, ntoken, embed, nhead, hidden, nlayers, dropout=0.5):
        super(ChatbotModel, self).__init__()
        self.transformer = Transformer(embed, nhead, dim_feedforward=hidden, num_encoder_layers=nlayers, num_decoder_layers=nlayers, dropout=dropout)
        self.pos_encoder = PositionalEncoding(embed, dropout)
        self.encoder = nn.Embedding(ntoken, embed)

        self.pos_encoder_d = PositionalEncoding(embed, dropout)
        self.encoder_d = nn.Embedding(ntoken, embed)

        self.embed = embed
        self.ntoken = ntoken
        self.linear = nn.Linear(embed, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, srcmask, tgtmask, srcpadmask, tgtpadmask):
        src = self.encoder(src) * math.sqrt(self.embed)
        src = self.pos_encoder(src)

        tgt = self.encoder_d(tgt) * math.sqrt(self.embed)
        tgt = self.pos_encoder_d(tgt)

        output = self.transformer(src.transpose(0,1), tgt.transpose(0,1), srcmask, tgtmask, src_key_padding_mask=srcpadmask, tgt_key_padding_mask=tgtpadmask)
        output = self.linear(output)
        return output


def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence


def train():
    MAX_LENGTH = 40
    BATCH_SIZE = 64
    lr = 1e-4
    epoch = 2
    vocab_size = 8000


    pre = Preprocess()
    questions, answers = pre.remove_punctuation()

    # csv를 하나씩 한 줄씩 잘라서 txt로 바꿔줌
    with open('all.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(questions))
        f.write('\n'.join(answers))

    vocab = pre.sent_piece("all.txt")

    questions_encode, answers_encode = pre.tokenize_and_filter(questions, answers, MAX_LENGTH, vocab)
    print("Question encoded example: ", questions_encode[0])
    print("Answer encoded example: ", answers_encode[0])

    dataset = SequenceDataset(questions_encode, answers_encode)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

    model = ChatbotModel(vocab_size+7, 256, 8, 512, 2, 0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for i in range(epoch):
        batchloss = 0.0

        progress = tqdm(dataloader)
        for (inputs, dec_inputs, outputs) in progress:
            optimizer.zero_grad()
            src_mask = model.generate_square_subsequent_mask(MAX_LENGTH).to(device)
            src_padding_mask = gen_attention_mask(inputs).to(device)
            tgt_mask = model.generate_square_subsequent_mask(MAX_LENGTH-1).to(device)
            tgt_padding_mask = gen_attention_mask(dec_inputs).to(device)

            result = model(inputs.to(device), dec_inputs.to(device), src_mask,
                           tgt_mask, src_padding_mask, tgt_padding_mask)
            loss = criterion(result.permute(1, 2, 0), outputs.to(device).long())
            progress.set_description("{:0.3f}".format(loss))
            loss.backward()
            optimizer.step()
            batchloss += loss
        print("epoch:", i + 1, "|", "loss:", batchloss.cpu().item() / len(dataloader))

    return vocab, model


def evaluate(sentence, vocab, model):
    START_TOKEN = [2]
    END_TOKEN = [3]
    MAX_LENGTH = 40

    sentence = preprocess_sentence(sentence)
    input = torch.tensor([START_TOKEN + vocab.encode_as_ids(sentence) + END_TOKEN]).to(device)
    output = torch.tensor([START_TOKEN]).to(device)

    # 디코더의 예측 시작
    model.eval()
    for i in range(MAX_LENGTH):
        src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(device)
        tgt_mask = model.generate_square_subsequent_mask(output.shape[1]).to(device)

        src_padding_mask = gen_attention_mask(input).to(device)
        tgt_padding_mask = gen_attention_mask(output).to(device)

        predictions = model(input, output, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask).transpose(0, 1)
        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = torch.LongTensor(torch.argmax(predictions.cpu(), axis=-1))


        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if torch.equal(predicted_id[0][0], torch.tensor(END_TOKEN[0])):
            break

        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        output = torch.cat([output, predicted_id.to(device)], axis=1)

    return torch.squeeze(output, axis=0).cpu().numpy()


def predict(sentence, vocab, model):
    vocab_size = 8000
    prediction = evaluate(sentence, vocab, model)
    predicted_sentence = vocab.Decode(list(map(int, [i for i in prediction if i < vocab_size + 7])))

    # print('Input: {}'.format(sentence))
    print('Chatbot say: {}'.format(predicted_sentence))
    print()

    return predicted_sentence

if __name__ == '__main__':
    vocab, model = train()
    data = pd.read_csv("KETI_대화데이터_응급상황_QA.csvㅇㅇㄹㅇ")

    is_init = True
    # 입력 받기
    while True:
        if is_init:
            idx = np.random.randint(0, len(data)-1, size=1)
            print("Chatbot: ", data['Q'].iloc[idx])
        query = input("You say: ")

        if query == "exit":
            break

        result = predict(query, vocab, model)

        print("Chatbot: ", result)




