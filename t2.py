import numpy as np
import os
import tensorflow as tf
import re
import tensorflow_text as tf_text

def read_raw_dataset_texts_fn():

    raw_txt_file = os.path.join(os.environ['spanish_english'], 'spa.txt')
    with open(raw_txt_file, 'rt', encoding='utf-8') as fh:
        lines = fh.read().splitlines()

    return lines


raw_lines = read_raw_dataset_texts_fn()
raw_lines = [line.split('\t') for line in raw_lines]
n_lines = len(raw_lines)
english_raw_lines = [line[0] for line in raw_lines]
spanish_raw_lines = [line[1] for line in raw_lines]

punctuations = '.?!,¿'
match_pattern = f'[{punctuations}]'
replace_pattern = r' \0 '

def count_words_fn(english_line):
    t0 = tf.strings.regex_replace(english_line, match_pattern, replace_pattern)
    t0 = t0.numpy()
    t0 = t0.decode('utf-8')
    t0 = t0.split()
    t0 = len(t0)

    return t0

english_lens = [count_words_fn(line) for line in english_raw_lines]
spanish_lens = [count_words_fn(line) for line in spanish_raw_lines]

english_lens = np.asarray(english_lens)
spanish_lens = np.asarray(spanish_lens)
english_idx = english_lens.argmax()
spanish_idx = spanish_lens.argmax()
print()








