import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
Position Encoding
unit module 은 나중에 다시 보기.
우선 큰 틀에 집중.
"""
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))

        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position = tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        # 짝수 인덱스(2i)에 사인 함수 적용
        sines = tf.math.sin(angle_rads[:, 0::2])

        # 홀수 인덱스(2i+1)에 코사인 함수 적용
        consine = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = consine
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    # position encoding의 리턴값은 3차원 텐서
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self):
        self.d_model = 512
        self.num_heads = 8
        self.d_depth = self.d_model // self.num_heads           # 64

        self.W_Q = tf.keras.layers.Dense(units = self.d_model)
        self.W_K = tf.keras.layers.Dense(units = self.d_model)
        self.W_V = tf.keras.layers.Dense(units = self.d_model)
        self.W_O = tf.keras.layers.Dense(units = self.d_model)

    def split_heads(self, input, batch_size):
        # 우선 하나만 구현해보자.
        ret = tf.reshape(input, shape=(batch_size, -1, self.num_heads, self.d_depth))          # (batch, -1, 8, 64)

        return tf.transpose(input, perm=[0, 2, 1, 3])

    def scaled_dot_attention(self, query, key, value, mask=None):                  # tf.keras.layers.Attention을 사용해도 됨.
        # query, key, value의 dimension: (batch_size, num_heads, seq_length, d_depth)
        depth = tf.cast(self.d_depth, tf.dtypes.float32)
        attn_score = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(depth)
        attn_score = tf.nn.softmax(attn_score, axis=-1)                       # sequence 기준 softmax

        ret = tf.matmul(attn_score, value)

        return ret

    # input은 query, key, value, mask 정보를 담은 dictionary
    def multihead_attention(self, input=None):
        query, key, value, mask = input['query'], input['key'], input['value'], input['mask']           # query = key = value in self-attention, not in inter-attention.

        query = self.W_Q(query)
        key = self.W_K(key)
        value = self.W_V(value)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        attn_scaled = self.scaled_dot_attention(query, key, value)
        attn_concat = tf.reshape(attn_scaled, shape=(attn_scaled.shape[0], -1, self.d_model))

        output = self.W_O(attn_concat)

        return output

    # input is scaled_dot_producted tensor (batch_size, seq_len, seq_len)
    def padding_mask(self, x):
        for i in range(x.shape[-1]):





class Transformer:
    def __init__(self):
        self.num_layers = 6
        self.d_ff = 2048
        self.d_model = 512

        self.encoder = self.encoding_layers()
        self.decoder = self.decoding_layers()

    def encoding_layers(self):


    def decoding_layers(self):

    def transformer(self, input):
        # input 받아서 encoder, decoder 각각 실행