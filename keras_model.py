import tensorflow as tf
import keras.api._v2.keras as keras
import numpy as np


class Encoder(keras.layers.Layer):

    def __init__(self, vocab_size, embedding_size, rnn_units, **kwargs):

        super(Encoder, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rnn_units = rnn_units

        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=self.embedding_size,
            mask_zero=False
        )
        with tf.device('/cpu'):
            self.embedding.build([None, 10])
        assert self.embedding.built
        assert 'cpu' in self.embedding.variables[0].device.lower()

        self.rnn = keras.layers.Bidirectional(
            merge_mode='sum',
            layer=keras.layers.GRU(
                units=rnn_units,
                return_state=False,
                return_sequences=True,
                recurrent_initializer='glorot_uniform',
                reset_after=True
            )
        )

    def call(self, spanish_tokens, spanish_token_mask):

        s = None
        spanish_tokens.set_shape([None, s])
        spanish_token_mask.set_shape([None, s])

        x = spanish_tokens
        input_mask = spanish_token_mask
        x = self.embedding(x)
        x.set_shape([None, s, self.embedding_size])

        x = self.rnn(x, mask=input_mask)
        x.set_shape([None, s, self.rnn_units])

        return x


class CrossAttention(keras.layers.Layer):

    def __init__(self, attention_units, **kwargs):

        super(CrossAttention, self).__init__(**kwargs)

        self.mha = keras.layers.MultiHeadAttention(
            num_heads=1,
            key_dim=attention_units
        )
        self.layer_norm = keras.layers.LayerNormalization(axis=-1)

    def call(self, spanish_rnn_states, english_rnn_states, spanish_mask, english_mask):

        s = None
        t = None
        f = None
        h = None

        spanish_rnn_states.set_shape([None, s, f])
        english_rnn_states.set_shape([None, t, f])
        spanish_mask.set_shape([None, s])
        english_mask.set_shape([None, t])

        from_eng_to_spa_mask = english_mask[:, :, None] & spanish_mask[:, None, :]
        from_eng_to_spa_mask.set_shape([None, t, s])

        attn_outputs, attn_weights = self.mha(
            query=english_rnn_states,
            key=spanish_rnn_states,
            value=spanish_rnn_states,
            attention_mask=from_eng_to_spa_mask,
            return_attention_scores=True
        )
        attn_weights.set_shape([None, h, t, s])
        attn_weights = tf.reduce_mean(attn_weights, axis=1)
        attn_weights.set_shape([None, t, s])
        attn_weights = tf.where(from_eng_to_spa_mask, attn_weights, 0.)

        attn_outputs.set_shape([None, t, f])
        attn_outputs = tf.where(english_mask[:, :, None], attn_outputs, 0.)
        attn_outputs = attn_outputs + english_rnn_states
        # here normalization is applied to axis -1, meaning we treat [None, t] as example dimensions.

        attn_outputs = self.layer_norm(attn_outputs)

        return dict(
            outputs=attn_outputs,
            weights=attn_weights
        )


class Decoder(keras.layers.Layer):

    def __init__(self, vocab_size, embedding_size, rnn_units, **kwargs):

        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rnn_units = rnn_units

        self.embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=embedding_size,
            mask_zero=False
        )
        with tf.device('/cpu'):
            self.embedding.build([None, 10])
        assert self.embedding.built
        assert 'cpu' in self.embedding.variables[0].device.lower()

        self.rnn = keras.layers.GRU(
            units=self.rnn_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
            reset_after=True
        )
        self.attention = CrossAttention(attention_units=self.rnn_units)
        self.output_dense = keras.layers.Dense(units=self.vocab_size)

    def call(self, spanish_rnn_states, spanish_token_mask, english_tokens_in, english_token_mask, english_rnn_initial_state=None):

        t = None
        s = None
        f = None

        spanish_rnn_states.set_shape([None, s, f])
        spanish_token_mask.set_shape([None, s])

        english_tokens_in.set_shape([None, t])
        english_token_mask.set_shape([None, t])

        english_embeddings = self.embedding(english_tokens_in)
        english_embeddings.set_shape([None, t, self.embedding_size])

        english_rnn_states, english_final_state = self.rnn(
            english_embeddings, initial_state=english_rnn_initial_state, mask=english_token_mask
        )
        english_rnn_states.set_shape([None, t, self.rnn_units])
        english_final_state.set_shape([None, self.rnn_units])

        english_rnn_states = self.attention(
            spanish_rnn_states=spanish_rnn_states,
            english_rnn_states=english_rnn_states,
            spanish_mask=spanish_token_mask,
            english_mask=english_token_mask
        )['outputs']
        english_rnn_states.set_shape([None, t, f])

        english_rnn_states = self.output_dense(english_rnn_states)

        return dict(
            states=english_rnn_states,
            final_state=english_final_state
        )


class Translator(keras.Model):

    def __init__(self, vocab_size, embedding_size, rnn_units,
                 max_seq_len, start_token_idx, end_token_idx,
                 **kwargs):

        super(Translator, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rnn_units = rnn_units

        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            rnn_units=rnn_units
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            rnn_units=rnn_units
        )

        self.max_seq_len = max_seq_len
        self.start_token_idx = tf.convert_to_tensor(start_token_idx, tf.int64)
        self.end_token_idx = tf.convert_to_tensor(end_token_idx, tf.int64)

    def call(self, spanish_tokens, spanish_token_mask, english_tokens_in, english_token_mask):

        s = None
        t = None

        spanish_tokens.set_shape([None, s])
        spanish_token_mask.set_shape([None, s])

        english_tokens_in.set_shape([None, t])
        english_token_mask.set_shape([None, t])

        spanish_rnn_states = self.encoder(spanish_tokens=spanish_tokens, spanish_token_mask=spanish_token_mask)
        spanish_rnn_states.set_shape([None, s, self.rnn_units])
        logits = self.decoder(
            spanish_rnn_states=spanish_rnn_states,
            spanish_token_mask=spanish_token_mask,
            english_tokens_in=english_tokens_in,
            english_token_mask=english_token_mask,
            english_rnn_initial_state=None
        )['states']
        logits.set_shape([None, t, self.vocab_size])

        return logits

    def translate(self, spanish_tokens, spanish_mask):

        s = None
        spanish_tokens.set_shape([None, s])
        spanish_mask.set_shape([None, s])

        spanish_rnn_states = self.encoder(
            spanish_tokens=spanish_tokens,
            spanish_token_mask=spanish_mask,
            training=False
        )
        spanish_rnn_states.set_shape([None, s, self.rnn_units])

        b = spanish_tokens.shape[0]
        english_tokens_in = tf.fill([b, 1], self.start_token_idx)
        english_mask = tf.fill([b, 1], True)
        is_done = tf.fill([b, 1], False)
        htm1 = None
        english_tokens_out_seq = []
        english_tokens_out_seq.append(english_tokens_in.numpy())
        for in_token_idx in range(self.max_seq_len):
            logits_htm1_dict = self.decoder(
                spanish_rnn_states=spanish_rnn_states,
                spanish_token_mask=spanish_mask,
                english_tokens_in=english_tokens_in,
                english_token_mask=english_mask,
                english_rnn_initial_state=htm1,
                training=False
            )
            logits = logits_htm1_dict['states']
            htm1 = logits_htm1_dict['final_state']
            english_tokens_out = tf.argmax(logits, axis=-1, output_type=tf.int64)

            english_tokens_in = english_tokens_out
            english_tokens_out_seq.append(english_tokens_in.numpy())
            english_mask = english_mask & (english_tokens_in != self.end_token_idx)
            is_done = is_done | (english_tokens_in == self.end_token_idx)
            should_exit = tf.reduce_all(is_done).numpy()
            if should_exit:
                break

        english_tokens_out = np.concatenate(english_tokens_out_seq, axis=1)

        return english_tokens_out


if __name__ == '__main__':

    model = Translator(
        vocab_size=5000,
        embedding_size=256,
        rnn_units=256
    )

    token_range = np.arange(1, 5000).astype(np.int32)
    spanish_tokens = np.random.choice(token_range, size=[64, 20])
    spanish_tokens = np.pad(spanish_tokens, [[0, 0], [0, 3]])
    english_tokens_in = np.random.choice(token_range, size=[64, 17])
    english_tokens_in = np.pad(english_tokens_in, [[0, 0], [0, 3]])

    spanish_mask = spanish_tokens > 0
    english_mask = english_tokens_in > 0

    spanish_tokens = tf.convert_to_tensor(spanish_tokens)
    spanish_mask = tf.convert_to_tensor(spanish_mask)

    english_tokens_in = tf.convert_to_tensor(english_tokens_in)
    english_mask = tf.convert_to_tensor(english_mask)

    logits = model(
        spanish_tokens=spanish_tokens,
        spanish_token_mask=spanish_mask,
        english_tokens_in=english_tokens_in,
        english_token_mask=english_mask
    )

    for idx, w in enumerate(model.variables):
        device = w.device
        if 'CPU' in device:
            device = 'CPU'
        elif 'GPU' in device:
            device = 'GPU'
        else:
            assert False
        print(idx, device, w.name, w.shape)




