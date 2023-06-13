import tensorflow as tf
import tensorflow_text as tf_text
from main import Config
import numpy as np
import keras.api._v2.keras as keras

text = '¿Todavía está en casa?'


def tf_lower_and_split_punct_fn(text):

    punctuations = '.?!,¿'
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, f'[^ a-z{punctuations}]', '')
    text = tf.strings.regex_replace(text, f'[{punctuations}]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')

    return text

lines = Config.read_raw_dataset_texts_fn()
pairs = [line.split('\t') for line in lines]
spanish_raw_sentences = np.asarray([p[1] for p in pairs])
english_raw_sentences = np.asarray([p[0] for p in pairs])

spanish_tokenization = keras.layers.TextVectorization(
    max_tokens=5000,
    standardize=tf_lower_and_split_punct_fn,
    ragged=True
)
english_tokenization = keras.layers.TextVectorization(
    max_tokens=5000,
    standardize=tf_lower_and_split_punct_fn,
    ragged=True
)

english_tokenization.adapt(english_raw_sentences)
spanish_tokenization.adapt(spanish_raw_sentences)

for file_name, vocab in (('spanish_vocabulary_5000.txt', spanish_tokenization.get_vocabulary()),
                         ('english_vocabulary_5000.txt', english_tokenization.get_vocabulary())):

    with open(file_name, 'wt', encoding='utf-8') as fh:
        fh.write('\n'.join(vocab))

    with open(file_name, 'rt', encoding='utf-8') as fh:
        lines = fh.read().splitlines()

    assert vocab == lines
    print('done')



