import numpy as np

import typing
from typing import Any, Tuple

import einops

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf
import tensorflow_text as tf_text
import pathlib
import keras.api._v2.keras as keras

path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'

def load_data(path):
    text = path.read_text(encoding='utf-8')

    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]

    context = np.array([context for target, context in pairs])
    target = np.array([target for target, context in pairs])

    return target, context


english_sentences_raw, spanish_sentences_raw = load_data(path_to_file)
print(spanish_sentences_raw[-1])

n_sentences = len(english_sentences_raw)
BUFFER_SIZE = n_sentences
BATCH_SIZE = 64

training_example_indices = np.random.uniform(size=(n_sentences,)) < 0.8
train_raw = tf.data.Dataset.from_tensor_slices(
    dict(
        english=english_sentences_raw[training_example_indices],
        spanish=spanish_sentences_raw[training_example_indices]
    )
)
train_raw = train_raw.shuffle(buffer_size=n_sentences, reshuffle_each_iteration=True)
train_raw = train_raw.batch(batch_size=BATCH_SIZE)

t_indices = ~training_example_indices
val_raw = tf.data.Dataset.from_tensor_slices(
    dict(
        english=english_sentences_raw[t_indices],
        spanish=spanish_sentences_raw[t_indices]
    )
)
val_raw = val_raw.shuffle(buffer_size=n_sentences, reshuffle_each_iteration=True)
val_raw = val_raw.batch(batch_size=BATCH_SIZE)

for example_strings in train_raw.take(1):
    print(example_strings['spanish'][:5])
    print()
    print(example_strings['english'][:5])

example_text_utf8 = '¿Todavía está en casa?'
example_text = tf.convert_to_tensor(example_text_utf8)
t = tf_text.normalize_utf8(example_text, 'NFKD').numpy()
print(example_text.numpy())
print(t)

def tf_lower_and_split_punct(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

print(example_text.numpy().decode())
print(tf_lower_and_split_punct(example_text).numpy().decode())

max_vocab_size = 5000
spanish_text_processor = keras.layers.TextVectorization(
    max_tokens=max_vocab_size,
    standardize=tf_lower_and_split_punct,
    ragged=True
)

spanish_text_processor.adapt(train_raw.map(lambda t_dict:t_dict['spanish']))
t = spanish_text_processor.get_vocabulary()[:10]
print(t)

english_text_processor = keras.layers.TextVectorization(
    max_tokens=max_vocab_size,
    standardize=tf_lower_and_split_punct,
    ragged=True
)
english_text_processor.adapt(train_raw.map(lambda t_dict: t_dict['english']))
t = english_text_processor.get_vocabulary()[:10]
print(t)

print()
example_tokens = spanish_text_processor(example_strings['spanish'])
print(example_tokens[:3])

spanish_vocab = spanish_text_processor.get_vocabulary()
spanish_vocab = np.asarray(spanish_vocab)
tokens = spanish_vocab[example_tokens[0].numpy()]
t = ' '.join(tokens)
print(t)

plt.subplot(1, 2, 1)
t = example_tokens.to_tensor()
plt.pcolormesh(t)
plt.title('token ids')

plt.subplot(1, 2, 2)
plt.pcolormesh(t != 0)
plt.title('mask')

plt.savefig('ragged_tokens.png')
plt.close()

def process_text_fn(ds_dict):
    spanish = ds_dict['spanish']
    english = ds_dict['english']
    spanish = spanish_text_processor(spanish).to_tensor()
    english = english_text_processor(english)
    english_in = english[:, :-1].to_tensor()
    english_out = english[:, 1:].to_tensor()

    return dict(
        spanish=spanish,
        english_in=english_in,
        english_out=english_out
    )

train_ds = train_raw.map(process_text_fn, tf.data.AUTOTUNE)
val_ds = val_raw.map(process_text_fn, tf.data.AUTOTUNE)

for ex_dict in train_ds.take(1):
    spanish_tok = ex_dict['spanish']
    english_in = ex_dict['english_in']
    english_out = ex_dict['english_out']
    print('spanish')
    print(spanish_tok[0, :10].numpy())
    print('english in and out')
    print(english_in[0, :10].numpy())
    print(english_out[0, :10].numpy())


class ShapeChecker:

    def __init__(self):

        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):

        if not tf.executing_eagerly():
            return

        parsed = einops.parse_shape(tensor, names)
        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)

            if (broadcast and new_dim == 1):
                continue

            if old_dim is None:
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"shape mismatch for dimension '{name}'\n"
                                 f"  found: {new_dim}\n"
                                 f"  expected: {old_dim}\n"
                                 )


class Encoder(keras.layers.Layer):

    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units

        self.embedding = keras.layers.Embedding(self.vocab_size, units, mask_zero=True)

        self.rnn = keras.layers.Bidirectional(
            merge_mode='sum',
            layer=keras.layers.GRU(units,
                                   return_sequences=True,
                                   recurrent_initializer='glorot_uniform')
        )

    def call(self, x):
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch s')

        x = self.embedding(x)
        shape_checker(x, 'batch s units')

        x = self.rnn(x)
        shape_checker(x, 'batch s units')

        return x

    def convert_input(self, spanish):
        spanish = tf.convert_to_tensor(spanish)
        if spanish.ndim == 0:
            spanish = spanish[None]

        spanish = self.text_processor(spanish).to_tensor()
        spanish = self(spanish)

        return spanish


UNITS = 256
encoder = Encoder(text_processor=spanish_text_processor, units=UNITS)
ex_context = encoder(spanish_tok)

print(f'context tokens, shape (batch, s): {spanish_tok.shape}')
print(f'encoder output, shape (batch, s): {ex_context.shape}')


class CrossAttention(keras.layers.Layer):

    def __init__(self, units, **kwargs):

        super(CrossAttention, self).__init__()
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=1, key_dim=units, **kwargs
        )
        self.layernorm = keras.layers.LayerNormalization(axis=-1)  # treat (example, time_step) as independent examples
        self.add = keras.layers.Add()

    def call(self, x, context):

        shape_checker = ShapeChecker()
        shape_checker(x, 'b t f')
        shape_checker(context, 'b s f')

        english_mask = x._keras_mask
        spanish_mask = context._keras_mask

        from_eng_to_span_mask = english_mask[:, :, None] & spanish_mask[:, None, :]
        shape_checker(from_eng_to_span_mask, 'b t s')

        attn_outputs, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            attention_mask=from_eng_to_span_mask,
            return_attention_scores=True
        )
        shape_checker(x, 'b t f')
        shape_checker(attn_scores, 'b h t s')

        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        shape_checker(attn_scores, 'b t s')
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_outputs])
        x = self.layernorm(x)

        return x


attention_layer = CrossAttention(UNITS)

embed = keras.layers.Embedding(input_dim=english_text_processor.vocabulary_size(),
                               output_dim=UNITS, mask_zero=True)

ex_tar_embed = embed(english_in)
result = attention_layer(ex_tar_embed, ex_context)

print(f'context sequence shape - {ex_context.shape}')
print(f'target sequence shape - {ex_tar_embed.shape}')
print(f'attention result shape - {result.shape}')
print(f'attention weights shape - {attention_layer.last_attention_weights.shape}')

t = attention_layer.last_attention_weights[0].numpy()
t = np.sum(t, axis=1)
print()

attention_weights = attention_layer.last_attention_weights
mask = spanish_tok != 0
mask = mask.numpy()

plt.subplot(1, 2, 1)
plt.pcolormesh(attention_weights[:, 0, :])
plt.title('attention weights')

plt.subplot(1, 2, 2)
plt.pcolormesh(mask)
plt.title('mask')

plt.savefig('weights_mask.png')
plt.close()


class Decoder(keras.layers.Layer):

    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, text_processor, units):

        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='',
            oov_token='[UNK]'
        )
        self.id_to_word = keras.layers.StringLookup(
            invert=True,
            vocabulary=text_processor.get_vocabulary(),
            mask_token='',
            oov_token='[UNK]'
        )
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')

        self.units = units

        self.embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=units,
            mask_zero=True
        )
        self.rnn = keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        self.attention = CrossAttention(units)

        self.output_layer = keras.layers.Dense(units=self.vocab_size)

    def call(self, context, x, state=None, return_state=False):

        shape_checker = ShapeChecker()
        shape_checker(x, 'b t')
        shape_checker(context, 'b s f')

        x = self.embedding(x)
        shape_checker(x, 'b t f')

        x, state = self.rnn(x, initial_state=state)
        shape_checker(x, 'b t f')

        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights
        shape_checker(x, 'b t f')
        shape_checker(self.last_attention_weights, 'b t s')

        logits = self.output_layer(x)
        shape_checker(logits, 'b t target_vocab_size')

        if return_state:
            return logits, state
        else:
            return logits

    def get_initial_sate(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], tf.bool)
        embedded = self.embedding(start_tokens)

        return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    def tokens_to_text(self, tokens):

        words = self.id_to_word(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
        result = tf.strings.regex_replace(result, ' *\[END\] * $', '')

        return result

    def get_next_token(self, context, next_token, done, state, temperature=0.):

        logits, state = self.call(
            context=context,
            x=next_token,
            state=state,
            return_state=True
        )

        if temperature == 0.:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :] / temperature
            next_token = tf.random.categorical(logits, num_samples=1)

        done = done | (next_token == self.start_token)
        next_token = tf.where(done, tf.constant(0, tf.int64), next_token)

        return next_token, done, state


decoder = Decoder(text_processor=english_text_processor, units=UNITS)
logits = decoder(context=ex_context, x=english_in)
print(f'encoder output shape - {ex_context.shape}')
print(f'input target token shape - {english_in.shape}')
print(f'logits shape - {logits.shape}')
print()

next_token, done, state = decoder.get_initial_sate(ex_context)
tokens = []
for n in range(10):
    next_token, done, state = decoder.get_next_token(
        ex_context, next_token, done, state, temperature=1.
    )
    tokens.append(next_token)

tokens = tf.concat(tokens, axis=-1)

result = decoder.tokens_to_text(tokens)
print(result[:3].numpy())


class Translator(keras.Model):

    def __init__(self, units, context_text_processor, target_text_processor):

        super(Translator, self).__init__()
        encoder = Encoder(text_processor=context_text_processor, units=units)
        decoder = Decoder(text_processor=target_text_processor, units=units)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        return logits

    def translate(self, texts, *, max_length=50, temperature=0.):

        context = self.encoder.convert_input(texts)
        batch_size = tf.shape(texts)[0]

        tokens = []
        attention_weights = []
        next_token, done, state = self.decoder.get_initial_sate(context)

        for _ in range(max_length):

            next_token, done, state = self.decoder.get_next_token(context, next_token, done, state, temperature)
            tokens.append(next_token)
            attention_weights.append(self.decoder.last_attention_weights)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break
        tokens = tf.concat(tokens, axis=-1)
        self.last_attention_weights = tf.concat(attention_weights, axis=1)

        result = self.decoder.tokens_to_text(tokens)

        return result


def masked_loss_fn(y_true, y_pred):

    y_true = tf.cast(y_true, tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    mask = y_true > 0
    loss = tf.where(mask, loss, 0.)
    loss = tf.reduce_sum(loss)
    n = tf.math.count_nonzero(mask, dtype=tf.int32)
    n = tf.cast(n, tf.float32)
    loss = loss / n

    return loss


def masked_acc_fn(y_true, y_pred):

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    match = y_true == y_pred
    mask = y_true > 0
    n = tf.math.count_nonzero(mask, dtype=tf.int32)
    n = tf.cast(n, tf.float32)
    match = mask & match
    match = tf.math.count_nonzero(match, dtype=tf.int32)
    match = tf.cast(match, tf.float32)

    acc = match / n

    return acc


model = Translator(
    units=UNITS,
    context_text_processor=spanish_text_processor,
    target_text_processor=english_text_processor
)

logits = model((spanish_tok, english_in))
print('translator')
print(f'context tokens shape - {spanish_tok.shape}')
print(f'target tokens shape - {english_in.shape}')
print(f'logits shape - {logits.shape}')

model.compile(
    optimizer='adam',
    loss=masked_loss_fn,
    metrics=[masked_acc_fn, masked_loss_fn]
)

def map_ds_to_keras_ds_fn(batch_dict):

    return (batch_dict['spanish'], batch_dict['english_in']), batch_dict['english_out']



keras_val_ds = val_ds.map(map_ds_to_keras_ds_fn)
vocab_size = 1. * english_text_processor.vocabulary_size()
expected_loss = np.log(vocab_size)
expected_acc = 1. / vocab_size
print('expected loss - {}'.format(expected_loss))
print('expected acc  - {}'.format(expected_acc))

model.evaluate(keras_val_ds, steps=20, return_dict=True)

keras_train_ds = train_ds.map(map_ds_to_keras_ds_fn)
history = model.fit(
    keras_train_ds.repeat(),
    epochs=20,
    steps_per_epoch=20,
    validation_data=keras_val_ds,
    validation_steps=20,
    callbacks=[keras.callbacks.EarlyStopping(patience=3)]
)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/token')
plt.legend()
plt.savefig('loss.png')
plt.close()

plt.plot(history.history['masked_acc_fn'], label='accuracy')
plt.plot(history.history['val_masked_acc_fn'], label='val_accuracy')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/token')
plt.legend()
plt.savefig('acc.png')
plt.close()

result = model.translate(['¿Todavía está en casa?']) # Are you still home
t = result[0].numpy().decode()
print(t)




















