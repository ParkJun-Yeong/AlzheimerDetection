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


    def scaled_dot_attention(self, query, key, value, mask=None):                  # tf.keras.layers.Attention을 사용해도 됨.
        # query, key, value의 dimension: (batch_size, num_heads, seq_length, d_depth)
        depth = tf.cast(self.d_depth, tf.dtypes.float32)
        attn_scaled = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(depth)

        if mask is not None:
            attn_scaled += (mask * -1e9)

        attn_score = tf.nn.softmax(attn_scaled, axis=-1)                       # sequence 기준 softmax

        ret = tf.matmul(attn_score, value)

        return ret

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self):
        self.d_model = 512
        self.num_heads = 8
        self.d_depth = self.d_model // self.num_heads           # 64

        self.W_Q = tf.keras.layers.Dense(units = self.d_model)
        self.W_V = tf.keras.layers.Dense(units = self.d_model)
        self.W_O = tf.keras.layers.Dense(units = self.d_model)

    def split_heads(self, input, batch_size):
        # 우선 하나만 구현해보자.
        ret = tf.reshape(input, shape=(batch_size, -1, self.num_heads, self.d_depth))          # (batch, -1, 8, 64)

        return tf.transpose(input, perm=[0, 2, 1, 3])

    # input은 query, key, value, mask 정보를 담은 dictionary
    def call(self, input=None):
        query, key, value, mask = input['query'], input['key'], input['value'], input['mask']           # query = key = value in self-attention, not in inter-attention.

        # Query, Key, Value 생성
        query = self.W_Q(query)
        key = self.W_K(key)
        value = self.W_V(value)

        # Multihead 로 나누기
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # scaled dot attention
        # output: (batch_size, num_heads, seq_len, d_depth)
        attn_scaled = self.scaled_dot_attention(query, key, value)
        # reshape output shape to (batch_size, seq_len, d_model) (transpose를 하는 이유는 seq_len을 유지하기 위해서)
        attn_scaled = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        attn_concat = tf.reshape(attn_scaled, shape=(attn_scaled.shape[0], -1, self.d_model))

        output = self.W_O(attn_concat)

        return output


class Transformer:
    def __init__(self):
        self.num_layers = 6
        self.d_ff = 2048
        self.d_model = 512

        self.encoder = None
        self.decoder = None

        # softmax로 넘기기 전.
        # 매우 작은 수 1e-9로 변경.

    def create_padding_mask(self, x):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)

        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, x):
        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # 대각선도 빼려고 이렇게 함.
        padding_mask = self.create_padding_mask(x)  # padding mask 도 생성.

        return tf.maximum(look_ahead_mask, padding_mask)  # 어차피 둘 다 masking 해야 하므로 1인거 전부 가져오기.

    # 하나의 인코더 층
    def encoder_layer(self, dropout, name="encoder_layer"):
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")

        padding_mask = tf.keras.Input(shape=(1,1,None), name="padding_mask")

        attention = MultiHeadAttention(name="encoder_self_attention")({
            'query': inputs, 'key': inputs, 'value': inputs,
            'mask': padding_mask
        })

        # dropout + residual_net + layer normalization
        attention = tf.keras.layers.Dropout(rate=dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

        # dropout + position-wise FFNN (second sublayer)
        outputs = tf.keras.layers.Dense(units=self.d_ff, activation='relu')(attention)
        outputs = tf.keras.layers.Desne(units=self.d_model)(outputs)

        # dropout + residual_net + layer normalization
        outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

        return tf.keras.Model(                                              # tf.keras.Model groups layers into an object with training and inference features.
            inputs=[inputs, padding_mask], outputs=outputs, name=name
        )

    def encoder(self, vocab_size, dff, dropout, name="encoder"):
        inputs = tf.keras.Input(shape=(None,), name="inputs")

        # encoder: padding mask, decoder: lookahead mask
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        # word embedding + positional enbdoing
        embeddings = tf.keras.layers.Embedding(vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(vocab_size, self.d_model)(embeddings)
        outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)


        # 인코더를 6번 수행 --> 6-stack
        for i in range(self.num_layers):
            outputs = encdoer_layer(dropout, name="encoder_layer_{}".format(i),
                                    )([outputs, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)


    def decoder_layer(self, dropout, name="decoer_layer"):
        inputs = tf.keras.Input(shape=(None, self.d_model), name="decoder_layer")
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name="encoder_outputs")

        # look-ahead mask
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name="look_ahead_mask")

        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        self_attention = MultiHeadAttention(name="decoder_self_attention")(inputs={
            'query': inputs, 'key': inputs, 'value': inputs,
            'mask': look_ahead_mask
        })

        self_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(self_attention + inputs)

        inter_attention = MultiHeadAttention(name="decoder_inter_attention")(inputs={
            'query': self_attention, 'key': enc_outputs, 'value': enc_outputs,
            'mask': padding_mask
        })
        inter_attention = tf.keras.layers.Dropout(rate=dropout)(inter_attention)
        inter_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inter_attention + self_attention)

        # position-wise FFNN
        output = tf.keras.layers.Dense(units=self.d_ff, activation='relu')(inter_attention)
        output = tf.keras.layers.Dense(units=output)(output)
        output = tf.keras.layers.Dropout(rate=dropout)(output)
        output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(output + inter_attention)

        return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=output, name=name)

    def decoder(self, vocab_size, dropout, name='decoder'):
        inputs = tf.keras.Input(shape=(None,), name='inputs')
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name='encoder_outputs')

        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name="look_ahead_mask")

        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        embeddings = tf.keras.layers.Embedding(vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(vocab_size, self.d_model)(embeddings)
        outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

        for i in range(self.num_layers):
            outputs = decoder_layer(dropout=dropout, name='decoder_layer_{}'.format(i),
                                    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

            return tf.keras.Model(
                inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
                outputs=outputs,
                name=name
            )

    def transformer(self, vocab_size, dropout, name='transformer'):
        enc_inputs = tf.keras.Input(shape=(None,), name='enc_inputs')
        dec_inputs = tf.keras.Input(shape=(None,), name='dec_inputs')

        enc_padding_mask = tf.keras.layers.Lambda(
            self.create_padding_mask, output_shape=(1, 1, None),
            name='enc_padding_mask')(enc_inputs)

        look_ahead_mask = tf.keras.layers.Lambda(
            self.create_look_ahead_mask, output_shape=(1, None, None),
            name='look_ahead_mask')(dec_inputs)

        dec_padding_mask = tf.keras.layers.Lambda(
            self.create_padding_mask, output_shape=(1, 1, None),
            name='dec_padding_mask')(enc_inputs)

        enc_outputs = encoder(vocab_size=vocab_size, dropout=dropout,
                              )(inputs=[enc_inputs, enc_padding_mask])

        dec_outputs = decoder(vocab_size=vocab_size, dropout=dropout,
                              )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

        outputs = tf.keras.layers.Dense(units=vocab_size, name='outputs')(dec_outputs)

        return tf.keras.Model(inputs=[enc_inputs, dec_inputs], outputs=outputs, name=name)


    def loss_function(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logit=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)