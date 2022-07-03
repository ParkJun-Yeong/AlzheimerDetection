import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow_datasets as tfds
import tensorflow as tf
import Transformer


class TransformerChatbot:
    def __init__(self):
        self.START_TOKEN = None
        self.END_TOKEN = None
        self.tokenizer = None

        self.VOCAB_SIZE = None
        self.MAX_LENGTH = 40
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 20000
        self.DROPOUT = 0.1
        self.EPOCH = 50


    def remove_punctuation(self, data):
        questions = []
        for sentence in data['Q']:
            sentence = re.sub(r"([?.!,])", r" \1", sentence)  # () is metacharacter of grouping.
            sentence = sentence.strip()
            questions.append(sentence)

        answers = []
        for sentence in data['A']:
            sentence = re.sub(r"([?.!,])", r" \1", sentence)  # () is metacharacter of grouping.
            sentence = sentence.strip()  # remove whitespace of start/end.
            answers.append(sentence)

            return questions, answers

    def make_vocabulary(self, questions, answers):
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            questions + answers, target_vocab_size=2 ** 13)

        # <SOS>, <EOS> 토큰도 vocabulary에 추가

        self.START_TOKEN, self.END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
        self.VOCAB_SIZE = tokenizer.vocab_size + 2

        print('<SOS> token no.: ', self.START_TOKEN)
        print('<EOS> token no.: ', self.END_TOKEN)
        print('단어 집합 크기: ', self.VOCAB_SIZE)

    def tokenizer_and_filter(inputs, outputs):
        tokenized_inputs, tokenized_outputs = [], []

        for (sentence1, sentence2) in zip(inputs, outputs):
            sentence1 = self.START_TOKEN + self.tokenizer.encode(sentence1) + END_TOKEN
            sentence2 = self.START_TOKEN + self.tokenizer.encode(sentence2) + END_TOKEN

            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentenc2)

        # padding
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=self.MAX_LENGTH, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=self.MAX_LENGTH, padding='post')

        return tokenized_inputs, tokenized_outputs

    def make_batch(self, questions, answers):
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': questions,
                'dec_inputs': answer[:, :-1]
            },
            {
                'outputs': answers[:, 1:]
            },
        ))

        dataset = dataset.cache()
        dataset = dataset.shuffle(self.BUFFER_SIZE)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def accuracy(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    def call(self):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
        train_data = pd.read_csv('ChatBotData.csv')             # test data

        print("# number of samples: ", len(train_data))

        # if train_data.isnull().sum():
        #     # null 처리 코드
        #     print(train_data)

        # preprocessing
        questions, answers = self.remove_punctuation(train_data)

        # 단어 집합 생성
        self.make_vocabulary(questions, answers)
        questions, answers = self.tokenizer_and_filter(questions, answers)

        # 데이터셋 생성
        dataset = self.make_batch(questions, answers)

        tf.keras.backend.clear_session()

        transformer = Transformer()
        model = transformer.transformer(vocab_size=self.VOCAB_SIZE, dropout=self.DROPOUT)
        learning_rate = CustomSchedule(d_model=256)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        model.compile(optimizer=optimizer, loss=loss_function, metrics=[self.accuracy])

        model.fit(dataset, epochs=self.EPOCH)

    # Evaluation
    # def preprocess_sentence(self, sentence):
    #     sentence


if __name__ == '__main__':
    TransformerChatbot()
