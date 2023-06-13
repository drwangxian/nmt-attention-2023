

DEBUG = False

import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import tensorflow as tf
tf.random.set_seed(2023)
import numpy as np
np.random.seed(2023)
from argparse import Namespace
import os
import glob
from keras_model import Translator as KerasTranslator
import tensorflow_text as tf_text
import keras.api._v2.keras as keras


if DEBUG:
    for name in logging.root.manager.loggerDict:
        if name.startswith('numba'):
            logger = logging.getLogger(name)
            logger.setLevel(logging.WARNING)

        if name.startswith('matplotlib'):
            logger = logging.getLogger(name)
            logger.setLevel(logging.WARNING)


class Config:

    def __init__(self):

        self.debug_mode = DEBUG
        Config.set_gpu_fn()

        self.train_or_inference = Namespace(
            inference='d0-7',
            from_ckpt=None,
            ckpt_prefix=None
        )
        self.tb_dir = 'tb_inf'
        self.inferencing = self.train_or_inference.inference is not None

        if not self.inferencing:
            self.model_names = ('training', 'validation')
        else:
            self.model_names = ('training', 'validation', 'test')

        self.batch_size = 64
        self.vocab_size = 5000
        self.max_sentence_len = 55
        """
        excluding [START] and [END], maximum number of words(including punctuation marks):
        english: 49
        spanish: 51
        """

        self.initial_learning_rate = 1e-3
        self.patience_epochs = 5
        self.batches_per_epoch = None

        raw_lines = Config.read_raw_dataset_texts_fn()
        n_lines = len(raw_lines)
        self.dataset_and_tools_ins = SentenceTokenization(raw_lines)
        self.tvt_split_dict = Config.gen_tvt_partition_fn(n_lines)
        if self.debug_mode:
            for k in self.tvt_split_dict:
                self.tvt_split_dict[k] = self.tvt_split_dict[k][:10 * self.batch_size]

        self.demo_dataset_ins = TranslateDemo(self)

        self.translator_ins = Translator(self)

        if not self.inferencing:
            self.initialize_optimizer_fn()

    def initialize_optimizer_fn(self):

        self.learning_rate_tf_var = tf.Variable(
            self.initial_learning_rate, trainable=False, dtype=tf.float32, name='learning_rate_tf_var'
        )

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate_tf_var)

        batch_size = self.batch_size
        max_sen_len = self.max_sentence_len

        token_range = np.arange(10, 4990)
        spanish_tokens = np.random.choice(token_range, size=[batch_size, max_sen_len - 5])
        english_in_tokens = np.random.choice(token_range, size=[batch_size, max_sen_len - 3])
        english_out_tokens = np.random.choice(token_range, size=[batch_size, max_sen_len - 3])

        spanish_tokens = np.pad(spanish_tokens, [[0, 0], [0, 5]])
        english_in_tokens = np.pad(english_in_tokens, [[0, 0], [0, 3]])
        english_out_tokens = np.pad(english_out_tokens, [[0, 0], [0, 3]])
        english_mask = english_in_tokens > 0
        spanish_mask = spanish_tokens > 0

        spanish_tokens = tf.convert_to_tensor(spanish_tokens, tf.int64)
        spanish_mask = tf.convert_to_tensor(spanish_mask, tf.bool)
        english_in_tokens = tf.convert_to_tensor(english_in_tokens, tf.int64)
        english_out_tokens = tf.convert_to_tensor(english_out_tokens, tf.int64)
        english_mask = tf.convert_to_tensor(english_mask, tf.bool)

        translator_model = self.translator_ins.keras_translator_model

        with tf.GradientTape() as tape:
            logits = translator_model(
                spanish_tokens=spanish_tokens,
                spanish_token_mask=spanish_mask,
                english_tokens_in=english_in_tokens,
                english_token_mask=english_mask
            )
            loss = self.translator_ins.loss_tf_fn(labels=english_out_tokens, logits=logits, mask=english_mask)
        trainables = translator_model.trainable_variables
        assert len(trainables) > 2
        grads = tape.gradient(loss, trainables)
        assert all(g is not None for g in grads)
        self.optimizer.apply_gradients(zip(grads, trainables))
        assert len(self.optimizer.weights) > 1

    def chk_if_tb_dir_and_model_with_same_prefix_exist_fn(self):

        # check if tb_dir exists
        assert self.tb_dir is not None
        is_tb_dir_exist = glob.glob('{}/'.format(self.tb_dir))
        if is_tb_dir_exist:
            assert False, 'directory {} already exists'.format(self.tb_dir)

        # check if model exists
        if self.train_or_inference.inference is None and self.train_or_inference.ckpt_prefix is not None:
            ckpt_dir, ckpt_prefix = os.path.split(self.train_or_inference.ckpt_prefix)
            assert ckpt_prefix != ''
            if ckpt_dir == '':
                ckpt_dir = 'ckpts'

            is_exist = glob.glob('{}/{}*'.format(ckpt_dir, ckpt_prefix))
            if is_exist:
                assert False, 'checkpoints with prefix {} already exist'.format(ckpt_prefix)

    @staticmethod
    def set_gpu_fn():

        gpus = tf.config.list_physical_devices('GPU')
        num_gpus = len(gpus)
        assert num_gpus == 1
        tf.config.experimental.set_memory_growth(gpus[0], True)

    @staticmethod
    def read_raw_dataset_texts_fn():

        raw_txt_file = os.path.join(os.environ['spanish_english'], 'spa.txt')
        with open(raw_txt_file, 'rt', encoding='utf-8') as fh:
            lines = fh.read().splitlines()

        return lines

    @staticmethod
    def gen_tvt_partition_fn(n_lines):

        line_indices = np.arange(n_lines)
        np.random.shuffle(line_indices)
        t = np.asarray([ 24905,  48556,  40855, 113456, 117267,  51698,  81179,  84506,
                         5581,  58646])
        assert np.array_equal(t, line_indices[:10])
        n_training = int(np.round(n_lines * .8))
        n_validation = int(np.round(n_lines * 0.1))
        training_indices = line_indices[:n_training]
        validation_indices = line_indices[n_training:n_training + n_validation]
        test_indices = line_indices[n_training + n_validation:]

        training_indices = np.require(training_indices, np.int32, requirements=['O'])
        training_indices.flags['WRITEABLE'] = False
        validation_indices = np.require(validation_indices, np.int32, requirements=['O'])
        validation_indices.flags['WRITEABLE'] = False
        test_indices = np.require(test_indices, np.int32)
        test_indices.flags['WRITEABLE'] = False

        return dict(
            training=training_indices,
            validation=validation_indices,
            test=test_indices
        )


class Translator:

    def __init__(self, config):

        vocab_size = config.vocab_size
        rnn_units = 256
        embedding_size = 256

        dataset_and_tools_ins = config.dataset_and_tools_ins
        english_vocabulary = dataset_and_tools_ins.dataset_dict['english']['vocabulary']
        start_token_idx = english_vocabulary.index('[START]')
        end_token_idx = english_vocabulary.index('[END]')

        keras_translator_ins = KerasTranslator(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            rnn_units=rnn_units,
            start_token_idx=start_token_idx,
            end_token_idx=end_token_idx,
            max_seq_len=config.max_sentence_len
        )
        self.keras_translator_model = keras_translator_ins

        self.indices_to_words_fn = keras.layers.StringLookup(
            invert=True,
            vocabulary=english_vocabulary,
            mask_token='',
            oov_token='[UNK]'
        )

        self.model_for_ckpt = self.keras_translator_model
        self.config = config

    @tf.function(input_signature=[
        tf.TensorSpec([None, None], tf.int64, name='labels'),
        tf.TensorSpec([None, None, 5000], name='logits'),
        tf.TensorSpec([None, None], tf.bool, name='mask')
    ])
    def loss_tf_fn(self, labels, logits, mask):

        labels = tf.convert_to_tensor(labels, tf.int64)
        labels.set_shape([None, None])

        logits = tf.convert_to_tensor(logits, tf.float32)
        logits.set_shape([None, None, 5000])

        mask = tf.convert_to_tensor(mask, tf.bool)
        mask.set_shape([None, None])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.where(mask, loss, 0.)
        loss = tf.reduce_sum(loss)
        n = tf.math.count_nonzero(mask, dtype=tf.int32)
        n = tf.cast(n, tf.float32)
        loss = loss / n

        return loss

    def translate(self, spanish_tokens, spanish_mask):

        english_tokens = self.keras_translator_model.translate(
            spanish_tokens=spanish_tokens,
            spanish_mask=spanish_mask
        )
        english_sentences = self.indices_to_words_fn(english_tokens)

        return english_sentences


class SentenceTokenization:

    def __init__(self, raw_lines):

        with open(os.path.join(os.environ['spanish_english'], 'spanish_vocabulary_5000.txt')) as fh:
            spanish_vocab = fh.read().splitlines()
        spanish_tokenization = keras.layers.TextVectorization(
            max_tokens=5000,
            standardize=SentenceTokenization.tf_lower_and_split_punct_fn,
            ragged=True,
            vocabulary=spanish_vocab
        )

        with open(os.path.join(os.environ['spanish_english'], 'english_vocabulary_5000.txt')) as fh:
            english_vocab = fh.read().splitlines()
        english_tokenization = keras.layers.TextVectorization(
            max_tokens=5000,
            standardize=SentenceTokenization.tf_lower_and_split_punct_fn,
            ragged=True,
            vocabulary=english_vocab
        )

        raw_lines = [line.split('\t') for line in raw_lines]
        english_sentences = [line[0] for line in raw_lines]
        spanish_sentences = [line[1] for line in raw_lines]

        with tf.device('cpu'):
            english_tokenized_sentences_ragged_tensor = english_tokenization(english_sentences)
            spanish_tokenized_sentences_ragged_tensor = spanish_tokenization(spanish_sentences)

            english_mask = english_tokenized_sentences_ragged_tensor > 0
            spanish_mask = spanish_tokenized_sentences_ragged_tensor > 0

        self.dataset_dict = dict(
            english=dict(
                vocabulary=english_vocab,
                tokenizer=english_tokenization,
                sentences=english_sentences,
                sentence_lengths=SentenceTokenization.sentence_lens_fn(english_sentences),
                tokenized_ragged_tensor=english_tokenized_sentences_ragged_tensor,
                mask=english_mask
            ),
            spanish=dict(
                vocabulary=spanish_vocab,
                tokenizer=spanish_tokenization,
                sentences=spanish_sentences,
                sentence_lengths=SentenceTokenization.sentence_lens_fn(spanish_sentences),
                tokenized_ragged_tensor=spanish_tokenized_sentences_ragged_tensor,
                mask=spanish_mask
            )
        )

    @staticmethod
    def tf_lower_and_split_punct_fn(text):

        punctuations = '.?!,¿'
        text = tf_text.normalize_utf8(text, 'NFKD')
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, f'[^ a-z{punctuations}]', '')
        text = tf.strings.regex_replace(text, f'[{punctuations}]', r' \0 ')
        text = tf.strings.strip(text)
        text = tf.strings.join(['[START]', text, '[END]'], separator=' ')

        return text

    @staticmethod
    def sentence_lens_fn(sentences):

        assert isinstance(sentences, list)

        n_sens = len(sentences)
        lens = np.empty([n_sens], np.int32)
        for idx, sen in enumerate(sentences):
            _len = len(sen.split())
            lens[idx] = _len
        lens.flags['WRITEABLE'] = False

        return lens


class TFDataset:

    def __init__(self, model):

        self.model = model
        self.sentence_indices = model.config.tvt_split_dict[model.name]

        training_mode = not model.config.inferencing
        if training_mode:
            assert model.name != 'test'
        self.training_split_in_training_model = training_mode and model.name == 'training'

        dataset_and_tools = self.model.config.dataset_and_tools_ins.dataset_dict
        self.spanish_ragged_tensor = dataset_and_tools['spanish']['tokenized_ragged_tensor']
        self.spanish_mask_ragged_tensor = dataset_and_tools['spanish']['mask']
        self.english_ragged_tensor = dataset_and_tools['english']['tokenized_ragged_tensor']
        self.english_mask_ragged_tensor = dataset_and_tools['english']['mask']
        self.max_sen_len = self.model.config.max_sentence_len
        dataset_and_n_batches_dict = self.gen_tf_dataset_fn()
        self.tf_dataset = dataset_and_n_batches_dict['dataset']
        self.batches_per_iteration = dataset_and_n_batches_dict['batches_per_iteration']
        if self.training_split_in_training_model:
            self.iterator = iter(self.tf_dataset)
            assert model.config.batches_per_epoch is None
            model.config.batches_per_epoch = self.batches_per_iteration
            logging.info('batches per epoch set to {}'.format(self.batches_per_iteration))

    def map_batch_indices_to_data_fn(self, batch_indices):

        max_sen_len = self.max_sen_len

        spanish_batch = tf.gather(self.spanish_ragged_tensor, axis=0, indices=batch_indices)
        spanish_batch = spanish_batch.to_tensor()
        spanish_mask = tf.gather(self.spanish_mask_ragged_tensor, axis=0, indices=batch_indices)
        spanish_mask = spanish_mask.to_tensor()

        english_batch = tf.gather(self.english_ragged_tensor, axis=0, indices=batch_indices)
        english_batch = english_batch.to_tensor()
        english_mask = tf.gather(self.english_mask_ragged_tensor, axis=0, indices=batch_indices)
        english_mask = english_mask.to_tensor()

        spanish_batch = spanish_batch[:, :max_sen_len]
        spanish_mask = spanish_mask[:, :max_sen_len]
        english_batch = english_batch[:, :max_sen_len]
        english_mask = english_mask[:, :max_sen_len]

        english_in_batch = english_batch[:, :-1]
        english_out_batch = english_batch[:, 1:]
        english_mask = english_mask[:, :-1]

        spanish_batch.set_shape([None, None])
        spanish_mask.set_shape([None, None])
        english_in_batch.set_shape([None, None])
        english_out_batch.set_shape([None, None])
        english_mask.set_shape([None, None])

        return dict(
            spanish=spanish_batch,
            spanish_mask=spanish_mask,
            english_in=english_in_batch,
            english_out=english_out_batch,
            english_mask=english_mask
        )

    def gen_tf_dataset_fn(self):

        sentence_indices = tf.convert_to_tensor(self.sentence_indices, tf.int32)
        n_sentences = len(sentence_indices)
        batch_size = self.model.config.batch_size

        dataset = tf.data.Dataset.from_tensor_slices(sentence_indices)
        if self.training_split_in_training_model:
            dataset = dataset.shuffle(n_sentences, reshuffle_each_iteration=True)
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size)

        dataset = dataset.map(self.map_batch_indices_to_data_fn)

        batches_per_iteration = (n_sentences + batch_size - 1) // batch_size

        return dict(
            dataset=dataset,
            batches_per_iteration=batches_per_iteration
        )


class Metrics:

    def __init__(self, model):

        self.model = model
        self.tf_var_dict = self.define_tf_variables_fn()

        self.loss = None
        self.accuracy = None

    @staticmethod
    def tf_i4_divide_fn(numerator, denominator):

        numerator = tf.cast(numerator, tf.float32)
        denominator = tf.cast(denominator, tf.float32)

        return numerator / (denominator + 1e-7)

    def reset(self):

        for var in self.tf_var_dict.values():
            var.assign(tf.zeros_like(var))

        self.loss = None
        self.accuracy = None

    def define_tf_variables_fn(self):

        model = self.model

        with tf.name_scope(model.name):
            with tf.name_scope('statistics'):

                batch_counter = tf.Variable(
                    initial_value=tf.zeros([], tf.int32),
                    trainable=False,
                    name='batch_counter'
                )
                matches = tf.Variable(
                    initial_value=tf.zeros([], tf.int32),
                    trainable=False,
                    name='matches'
                )
                total = tf.Variable(
                    initial_value=tf.zeros([], tf.int32),
                    trainable=False,
                    name='total'
                )

                loss = tf.Variable(
                    initial_value=tf.zeros([], tf.float32),
                    trainable=False,
                    name='loss'
                )

                tf_var_dict = dict(
                    batch_counter=batch_counter,
                    matches=matches,
                    total=total,
                    loss=loss
                )

                return tf_var_dict

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None], tf.int64, name='labels'),
            tf.TensorSpec([None, None, 5000], name='logits'),
            tf.TensorSpec([None, None], tf.bool, name='mask'),
            tf.TensorSpec([], name='loss')
        ]
    )
    def _update_states_tf_fn(self, labels, logits, mask, loss):

        labels = tf.convert_to_tensor(labels, tf.int64)
        labels.set_shape([None, None])

        logits = tf.convert_to_tensor(logits, tf.float32)
        logits.set_shape([None, None, 5000])

        mask = tf.convert_to_tensor(mask, tf.bool)
        mask.set_shape([None, None])

        loss = tf.convert_to_tensor(loss, tf.float32)
        loss.set_shape([])

        self.tf_var_dict['loss'].assign_add(loss)
        self.tf_var_dict['batch_counter'].assign_add(1)

        logits = tf.argmax(logits, axis=-1, output_type=tf.int64)
        logits.set_shape([None, None])

        total = tf.math.count_nonzero(mask, dtype=tf.int32)
        self.tf_var_dict['total'].assign_add(total)
        matches = labels == logits
        matches = mask & matches
        matches = tf.math.count_nonzero(matches, dtype=tf.int32)
        self.tf_var_dict['matches'].assign_add(matches)

    def update_states(self, labels, logits, mask, loss):

        self._update_states_tf_fn(labels=labels, logits=logits, mask=mask, loss=loss)

    def results(self):

        vard = self.tf_var_dict

        tf.debugging.assert_greater(vard['batch_counter'], 0)
        tf.debugging.assert_greater(vard['total'], 0)

        loss = Metrics.tf_i4_divide_fn(vard['loss'], vard['batch_counter']).numpy()
        acc = Metrics.tf_i4_divide_fn(vard['matches'], vard['total']).numpy()

        self.loss = loss
        self.accuracy = acc

        return dict(
            loss=loss,
            accuracy=acc
        )


class TBSummary:

    def __init__(self, model):

        assert hasattr(model, 'metrics')

        self.model = model

        self.tb_path = os.path.join(model.config.tb_dir, model.name)
        self.tb_summary_writer = tf.summary.create_file_writer(self.tb_path)

    def write_tb_summary_fn(self, step_int):

        model = self.model

        assert isinstance(step_int, int)

        with tf.name_scope(model.name):
            with tf.name_scope('statistics'):
                result_dict = model.metrics.results()

                with self.tb_summary_writer.as_default(step_int):
                    for name, value in result_dict.items():
                        tf.summary.scalar(name, value)

    def close(self):
        self.tb_summary_writer.close()


class Model:

    def __init__(self, config, name):

        assert name in config.model_names
        self.config = config
        self.name = name

        inferencing = config.inferencing

        if not inferencing:
            assert name != 'test'

        self.tf_dataset = TFDataset(self)
        self.metrics = Metrics(self)
        self.tb_summary = TBSummary(self)


class TranslateDemo:

    def __init__(self, config):

        self.config = config

        self.dataset_dict = self.create_demo_dataset_fn()

    def create_demo_dataset_fn(self, ):

        config = self.config

        selected_sentence_dict = {}
        for model_name in config.model_names:
            t = self.select_short_medium_and_long_sentence_indices_fn(model_name)
            selected_sentence_dict[model_name] = t

        dataset_dict = config.dataset_and_tools_ins.dataset_dict
        spanish_dict = dataset_dict['spanish']
        english_dict = dataset_dict['english']

        demo_dataset_dict = {}
        for model_name in config.model_names:
            idx_dict = selected_sentence_dict[model_name]
            demo_dataset_dict[model_name] = {}
            for len_name in idx_dict:
                len_idx = idx_dict[len_name]
                spanish_sentence = spanish_dict['sentences'][len_idx]
                spanish_tokenized = spanish_dict['tokenized_ragged_tensor'][len_idx]
                spanish_mask = spanish_dict['mask'][len_idx]

                english_sentence = english_dict['sentences'][len_idx]
                demo_dataset_dict[model_name][len_name] = dict(
                    spanish_sentence=spanish_sentence,
                    spanish_tokens=spanish_tokenized,
                    spanish_mask=spanish_mask,
                    english_sentence=english_sentence
                )

        return demo_dataset_dict

    def select_short_medium_and_long_sentence_indices_fn(self, model_name):

        config = self.config

        sen_indices = config.tvt_split_dict[model_name]
        dataset_dict = config.dataset_and_tools_ins.dataset_dict

        spanish_dict = dataset_dict['spanish']
        spanish_sen_lens = spanish_dict['sentence_lengths']
        spanish_sen_lens = spanish_sen_lens[sen_indices]

        arg_sort_indices = np.argsort(spanish_sen_lens)
        n_sens = len(arg_sort_indices)
        middle = n_sens // 2
        short_idx, medium_idx, long_idx = arg_sort_indices[[0, middle, -1]]
        short_idx = sen_indices[short_idx]
        medium_idx = sen_indices[medium_idx]
        long_idx = sen_indices[long_idx]

        return dict(
            short=short_idx,
            medium=medium_idx,
            long=long_idx
        )

    @staticmethod
    def join_remove_to_np_fn(tokens):

        t_eng = tf.strings.reduce_join(tokens, separator=' ')
        t_eng = tf.strings.regex_replace(t_eng, '^ *\[START\] *', '')
        t_eng = tf.strings.regex_replace(t_eng, ' *\[END\] *$', '')
        t_eng = t_eng.numpy().decode('utf-8')

        return t_eng

    def free_translate_fn(self, spanish_sentence):

        config = self.config
        translate_fn = config.translator_ins.translate

        dataset_and_tools_ins = config.dataset_and_tools_ins
        spanish_dict = dataset_and_tools_ins.dataset_dict['spanish']

        spanish_tokens = spanish_dict['tokenizer'](spanish_sentence)
        mask = tf.ones_like(spanish_tokens, tf.bool)
        t_eng = translate_fn(spanish_tokens=spanish_tokens[None, :], spanish_mask=mask[None, :])[0]
        translated = TranslateDemo.join_remove_to_np_fn(t_eng)
        logging.info(
            f'spanish -  {spanish_sentence}\n'
            f'translated - {translated}'
        )

    def __call__(self, model_dict, step_int=None):

        assert step_int is not None
        assert isinstance(step_int, int)

        config = self.config
        translate_fn = config.translator_ins.translate
        dataset_dict = self.dataset_dict

        logging.info('demonstration ...')
        for model_name in dataset_dict:
            logging.info(model_name)
            model_dataset_dict = dataset_dict[model_name]

            model = model_dict[model_name]
            tb_summary_writer = model.tb_summary.tb_summary_writer

            with tf.name_scope(model.name):
                with tf.name_scope('demonstration'):
                        for len_type in model_dataset_dict:
                            spanish_tokens = model_dataset_dict[len_type]['spanish_tokens']
                            spanish_mask = model_dataset_dict[len_type]['spanish_mask']
                            english_sentence = model_dataset_dict[len_type]['english_sentence']
                            spanish_sentence = model_dataset_dict[len_type]['spanish_sentence']
                            t_eng = translate_fn(spanish_tokens=spanish_tokens[None, :], spanish_mask=spanish_mask[None, :])[0]
                            translated_english_sentence = TranslateDemo.join_remove_to_np_fn(t_eng)

                            logging.info(
                                '\n'
                                f'{len_type}\n'
                                f'{spanish_sentence}\n'
                                f'{english_sentence}\n'
                                f'translated - {translated_english_sentence}'
                            )

                            with tf.name_scope(len_type):
                                with tb_summary_writer.as_default(step_int):
                                    tf.summary.text('spanish_english_translated',
                                                    '\n\n'.join([spanish_sentence, english_sentence, translated_english_sentence])
                                                    )




def main():

    MODEL_DICT = {}
    MODEL_DICT['config'] = Config()
    for name in MODEL_DICT['config'].model_names:
        MODEL_DICT[name] = Model(config=MODEL_DICT['config'], name=name)

    aug_info = []
    config = MODEL_DICT['config']
    aug_info.append('tb dir - {}'.format(config.tb_dir))
    aug_info.append('debug mode - {}'.format(config.debug_mode))
    aug_info.append('batch size - {}'.format(config.batch_size))
    inferencing = config.inferencing
    if not inferencing:
        aug_info.append('num of batches per epoch - {}'.format(config.batches_per_epoch))
        aug_info.append('num of patience epochs - {}'.format(config.patience_epochs))
        aug_info.append('initial learning rate - {}'.format(config.initial_learning_rate))
        if config.train_or_inference.from_ckpt is not None:
            aug_info.append('resume training from ckpt - {}'.format(config.train_or_inference.from_ckpt))
    else:
        aug_info.append('inference with ckpt - {}'.format(config.train_or_inference.inference))
    aug_info = '\n\n'.join(aug_info)
    logging.info(aug_info)

    with MODEL_DICT['training'].tb_summary.tb_summary_writer.as_default():
        tf.summary.text('auxiliary_information', aug_info, step=0)

    def training_fn(global_step=None):

        assert not inferencing
        assert isinstance(global_step, int)

        model = MODEL_DICT['training']
        iterator = model.tf_dataset.iterator
        translator_ins = config.translator_ins
        keras_model = translator_ins.keras_translator_model
        loss_fn = translator_ins.loss_tf_fn
        metrics = model.metrics
        batches_per_epoch = config.batches_per_epoch
        optimizer = config.optimizer
        trainables = keras_model.trainable_variables

        metrics.reset()
        for batch_idx in range(batches_per_epoch):
            logging.debug(f'batch {batch_idx}/{batches_per_epoch}')

            batch = iterator.get_next()
            with tf.GradientTape() as tape:
                logits = keras_model(
                    spanish_tokens=batch['spanish'],
                    spanish_token_mask=batch['spanish_mask'],
                    english_tokens_in=batch['english_in'],
                    english_token_mask=batch['english_mask'],
                    training=True
                )
                loss = loss_fn(labels=batch['english_out'], logits=logits, mask=batch['english_mask'])
            metrics.update_states(
                labels=batch['english_out'],
                logits=logits,
                mask=batch['english_mask'],
                loss=loss
            )
            grads = tape.gradient(loss, trainables)
            optimizer.apply_gradients(zip(grads, trainables))
        model.tb_summary.write_tb_summary_fn(global_step)

        loss = metrics.loss
        accuracy = metrics.accuracy
        logging.info(f'{model.name} - step - {global_step} - loss - {loss} - accuracy - {accuracy}')

    def inference_fn(model_name, global_step=None):

        assert inferencing or model_name == 'validation'
        assert isinstance(global_step, int)

        model = MODEL_DICT[model_name]
        translator_ins = config.translator_ins
        keras_model = translator_ins.keras_translator_model
        loss_fn = translator_ins.loss_tf_fn
        assert not hasattr(model.tf_dataset, 'iterator')
        iterator = iter(model.tf_dataset.tf_dataset)
        metrics = model.metrics
        batches_per_epoch = model.tf_dataset.batches_per_iteration

        metrics.reset()
        for batch_idx in range(batches_per_epoch):
            batch = iterator.get_next()
            logits = keras_model(
                spanish_tokens=batch['spanish'],
                spanish_token_mask=batch['spanish_mask'],
                english_tokens_in=batch['english_in'],
                english_token_mask=batch['english_mask'],
                training=False
            )
            loss = loss_fn(labels=batch['english_out'], logits=logits, mask=batch['english_mask'])
            metrics.update_states(
                labels=batch['english_out'],
                logits=logits,
                mask=batch['english_mask'],
                loss=loss
            )
        batch = iterator.get_next_as_optional()
        assert not batch.has_value()

        model.tb_summary.write_tb_summary_fn(global_step)
        loss = metrics.loss
        accuracy = metrics.accuracy
        assert loss is not None
        assert accuracy is not None
        logging.info(f'{model.name} - step - {global_step} - loss - {loss} - accuracy - {accuracy}')

    if inferencing:
        ckpt_file = config.train_or_inference.inference
        ckpt_dir, ckpt_name = os.path.split(ckpt_file)
        if ckpt_dir == '':
            ckpt_dir = 'ckpts'
            ckpt_file = os.path.join(ckpt_dir, ckpt_name)
        ckpt = tf.train.Checkpoint(model=config.translator_ins.model_for_ckpt)
        status = ckpt.restore(ckpt_file)
        status.expect_partial()
        status.assert_existing_objects_matched()

        logging.info('inferencing ... ')
        for model_name in config.model_names:
            logging.info(model_name)
            inference_fn(model_name, global_step=0)
        config.demo_dataset_ins(model_dict=MODEL_DICT, step_int=0)
        for model_name in config.model_names:
            MODEL_DICT[model_name].tb_summary.close()
    elif config.train_or_inference.from_ckpt is not None:
        assert hasattr(config, 'optimizer')
        ckpt = tf.train.Checkpoint(
            model=config.translator_ins.model_for_ckpt,
            optimizer=config.optimizer
        )
        ckpt_file = config.train_or_inference.from_ckpt
        ckpt_dir, ckpt_name = os.path.split(ckpt_file)
        if ckpt_dir == '':
            ckpt_dir = 'ckpts'
            ckpt_file = os.path.join(ckpt_dir, ckpt_name)
        status = ckpt.restore(ckpt_file)
        assert status.assert_consumed()
        logging.info('reproducing results ...')
        model_name = 'validation'
        logging.info(model_name)
        inference_fn(model_name, global_step=0)
        best_accuracy = MODEL_DICT[model_name].metrics.accuracy
        assert best_accuracy is not None
        best_epoch = 0
        config.demo_dataset_ins(model_dict=MODEL_DICT, step_int=0)
    else:
        logging.info('training from scratch ...')
        best_accuracy = None

    # training
    if not inferencing:
        assert config.train_or_inference.ckpt_prefix is not None
        assert 'ckpt_manager' not in MODEL_DICT
        assert hasattr(config, 'optimizer')
        ckpt = tf.train.Checkpoint(
            model=config.translator_ins.model_for_ckpt,
            optimizer=config.optimizer
        )
        ckpt_dir, ckpt_prefix = os.path.split(config.train_or_inference.ckpt_prefix)
        assert ckpt_prefix != ''
        if ckpt_dir == '':
            ckpt_dir = 'ckpts'
        ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            directory=ckpt_dir,
            max_to_keep=1,
            checkpoint_name=ckpt_prefix
        )
        MODEL_DICT['ckpt_manager'] = ckpt_manager
        training_epoch = 1

        while True:

            logging.info('\nepoch - {}'.format(training_epoch))

            for model_name in config.model_names:
                logging.info(model_name)
                if model_name == 'training':
                    training_fn(training_epoch)
                elif model_name == 'validation':
                    inference_fn(model_name, training_epoch)
                else:
                    assert False

            valid_accuracy = MODEL_DICT['validation'].metrics.accuracy
            should_save = best_accuracy is None or valid_accuracy > best_accuracy
            if should_save:
                best_accuracy = valid_accuracy
                best_epoch = training_epoch
                save_path = MODEL_DICT['ckpt_manager'].save(checkpoint_number=training_epoch)
                logging.info('weights checkpointed to {}'.format(save_path))
                config.demo_dataset_ins(model_dict=MODEL_DICT, step_int=training_epoch)

            d = training_epoch - best_epoch
            if d >= config.patience_epochs:
                logging.info('training terminated at epoch {}'.format(training_epoch))
                break

            training_epoch = training_epoch + 1

        for model_name in config.model_names:
            MODEL_DICT[model_name].tb_summary.close()


if __name__ == '__main__':

    main()











