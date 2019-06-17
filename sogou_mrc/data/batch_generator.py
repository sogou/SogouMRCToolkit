import tensorflow as tf
from collections import OrderedDict
import numpy as np
import collections
import six
import logging

import multiprocessing
class BatchGenerator(object):
    def __init__(self, vocab, instances, batch_size=32, use_char=True, training=False,
                 additional_fields=None, feature_vocab=None,num_parallel_calls=0,shuffle_ratio=1.0):
        self.instances = instances
        self.vocab = vocab
        self.batch_size = batch_size
        self.use_char = use_char
        self.training = training
        self.shuffle_ratio = shuffle_ratio
        self.num_parallel_calls = num_parallel_calls if num_parallel_calls>0 else multiprocessing.cpu_count()/2

        if self.instances is None or len(self.instances) == 0:
            raise ValueError('empty instances!!')

        self.additional_fields = additional_fields if additional_fields is not None else list()
        self.feature_vocab = feature_vocab if feature_vocab is not None else dict()

        self.batch, self.init_op = self.build_input_pipeline()

        sess_conf = tf.ConfigProto()
        sess_conf.gpu_options.allow_growth = True
        self.session = tf.Session(config=sess_conf)
        self.session.run(tf.tables_initializer())

    def next(self):
        return self.session.run(self.batch)

    def init(self):
        self.session.run(self.init_op)

    def get_instance_size(self):
        return len(self.instances)

    def get_batch_size(self):
        return self.batch_size

    def get_instances(self):
        return self.instances

    @staticmethod
    def detect_input_type_shape(instance, additional_fields=None):
        instance_keys = instance.keys()
        fields = ['context_tokens', 'question_tokens', 'answer_start', 'answer_end']
        try:
            for f in fields:
                assert f in instance_keys
        except:
            raise ValueError('A instance should contain at least "context_tokens", "question_tokens", \
                             "answer_start", "answer_end" four fields!')
        
        if additional_fields is not None and isinstance(additional_fields, list):
            fields.extend(additional_fields)

        def get_type(value):
            if isinstance(value, six.string_types):
                return tf.string
            elif isinstance(value, bool):
                return tf.bool
            elif isinstance(value, int):
                return tf.int32
            elif isinstance(value, float):
                return tf.float32
            else:
                return None

        input_type = {}
        input_shape = {}

        for field in fields:
            if instance[field] is None:
                if field not in ('answer_start', 'answer_end'):
                    logging.warning('Data type of field "%s" not detected! Skip this field.', field)
                continue
            elif isinstance(instance[field], list):
                if len(instance[field]) == 0:
                    logging.warning('Data shape of field "%s" not detected! Skip this field.', field)
                    continue
                
                field_type = get_type(instance[field][0])
                if field_type is not None:
                    input_type[field] = field_type
                    input_shape[field] = tf.TensorShape([None])
                else:
                    logging.warning('Data type of field "%s" not detected! Skip this field.', field)
            else:
                field_type = get_type(instance[field])
                if field_type is not None:
                    input_type[field] = field_type
                    input_shape[field] = tf.TensorShape([])
                else:
                    logging.warning('Data type of field "%s" not detected! Skip this field.', field)

        return fields, input_type, input_shape

    def build_input_pipeline(self):
        input_fields, input_type_dict, input_shape_dict = \
            BatchGenerator.detect_input_type_shape(self.instances[0], self.additional_fields)

        # 1. Get data
        def make_generator():
            for instance in self.instances:
                new_dict = {k: instance[k] for k in input_fields}
                yield new_dict

        dataset = tf.data.Dataset.from_generator(make_generator,
                                                 {w: input_type_dict[w] for w in input_fields},
                                                 {w: input_shape_dict[w] for w in input_fields}
                                                 )

        # 2. Character extracting function
        def extract_char(token, default_value="<PAD>"):
            out = tf.string_split(token, delimiter='')
            out = tf.sparse.to_dense(out, default_value=default_value)
            return out

        # 3. Build look-up table from vocabulary
        # 3.1 Word look-up table
        word_vocab = self.vocab.get_word_vocab()
        word_table = tf.contrib.lookup.index_table_from_tensor(tf.constant(word_vocab), num_oov_buckets=1)
        # 3.2 Char look-up table
        if self.use_char:
            char_vocab = self.vocab.get_char_vocab()
            char_table = tf.contrib.lookup.index_table_from_tensor(tf.constant(char_vocab), num_oov_buckets=1)
        # 3.3 other feature look-up table
        if len(self.feature_vocab) > 0:
            feature_table = {}
            for feature_name, vocab in self.feature_vocab.items():
                feature_table[feature_name] = tf.contrib.lookup.index_table_from_tensor(tf.constant(vocab),
                                                                                        num_oov_buckets=1)

        # 4. Some preprocessing, including char extraction, lowercasing, length
        def transform_new_instance(instance):
            context_tokens = instance['context_tokens']
            question_tokens = instance['question_tokens']

            if self.use_char:
                context_char = extract_char(context_tokens)
                context_word_len = tf.strings.length(context_tokens)
                question_char = extract_char(question_tokens)
                instance['context_char'] = tf.cast(char_table.lookup(context_char), tf.int32)
                instance['question_char'] = tf.cast(char_table.lookup(question_char), tf.int32)

            if self.vocab.do_lowercase:
                lower_context_tokens = tf.py_func(lambda x: np.char.lower(x.astype(np.bytes_)).astype(x.dtype),
                                                  [context_tokens], tf.string)
                lower_question_tokens = tf.py_func(lambda x: np.char.lower(x.astype(np.bytes_)).astype(x.dtype),
                                                   [question_tokens], tf.string)
                lower_context_tokens.set_shape(context_tokens.get_shape())
                lower_question_tokens.set_shape(question_tokens.get_shape())
                instance['context_word'] = tf.cast(word_table.lookup(lower_context_tokens), tf.int32)
                instance['question_word'] = tf.cast(word_table.lookup(lower_question_tokens), tf.int32)
            else:
                instance['context_word'] = tf.cast(word_table.lookup(context_tokens), tf.int32)
                instance['question_word'] = tf.cast(word_table.lookup(question_tokens), tf.int32)

            instance['context_len'] = tf.size(context_tokens)
            instance['question_len'] = tf.size(question_tokens)
            if len(self.feature_vocab) > 0:
                for field in self.additional_fields:
                    for feature_name, table in feature_table.items():
                        if field.endswith(feature_name):
                            instance[field] = tf.cast(table.lookup(instance[field]), tf.int32)
                            break
            return instance
        dataset = dataset.map(lambda fields: transform_new_instance(fields))

        # 5. Shuffle and repeat
        if self.training:
            dataset = dataset.shuffle(int(len(self.instances)*self.shuffle_ratio))
        dataset = dataset.prefetch(self.batch_size)

        # 6. Padding and batching
        def build_padded_shape(output_shapes):
            padded_shapes = dict()
            for field, shape in output_shapes.items():
                field_dim = len(shape.as_list())
                if field_dim > 0:
                    padded_shapes[field] = tf.TensorShape([None] * field_dim)
                else:
                    padded_shapes[field] = tf.TensorShape([])
            return padded_shapes

        padded_shapes = build_padded_shape(dataset.output_shapes)
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes)

        # 7. Make iterator and output
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        output_dict = iterator.get_next()
        return output_dict, init_op
