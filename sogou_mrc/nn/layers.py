# coding: utf-8
import tensorflow as tf
from sogou_mrc.nn.ops import dropout, add_seq_mask
from collections import defaultdict
import tensorflow_hub as hub
from sogou_mrc.libraries import modeling
import os

VERY_NEGATIVE_NUMBER = -1e29


class Layer(object):
    _name_dict = defaultdict(int)

    def __init__(self, name=None):
        if name is None:
            name = "layer"

        self.name = name + "_" + str(self._name_dict[name] + 1)
        self._name_dict[name] += 1


class Highway(Layer):
    def __init__(self,
                 affine_activation=tf.nn.relu,
                 trans_gate_activation=tf.nn.sigmoid,
                 hidden_units=0,
                 keep_prob=1.0,
                 name="highway"):
        super(Highway, self).__init__(name)

        self.affine_activation = affine_activation
        self.trans_gate_activation = trans_gate_activation
        self.affine_layer = None
        self.trans_gate_layer = None
        self.dropout = Dropout(keep_prob)
        if hidden_units > 0:
            self.affine_layer = tf.keras.layers.Dense(hidden_units, activation=self.affine_activation)
            self.trans_gate_layer = tf.keras.layers.Dense(hidden_units, activation=self.trans_gate_activation)

    def __call__(self, x, training=True):
        if self.trans_gate_layer is None:
            hidden_units = x.shape.as_list()[-1]
            self.affine_layer = tf.keras.layers.Dense(hidden_units, activation=self.affine_activation)
            self.trans_gate_layer = tf.keras.layers.Dense(hidden_units, activation=self.trans_gate_activation)

        gate = self.trans_gate_layer(x)
        trans = self.dropout(self.affine_layer(x), training=training)
        return gate * trans + (1. - gate) * x


class Dropout(Layer):
    def __init__(self, keep_prob=1.0, name="dropout"):
        super(Dropout, self).__init__(name)
        self.keep_prob = keep_prob

    def __call__(self, x, training):
        return dropout(x, self.keep_prob, training)


class VariationalDropout(Layer):
    def __init__(self, keep_prob=1.0, name="variational_dropout"):
        super(VariationalDropout, self).__init__(name)
        self.keep_prob = keep_prob

    def __call__(self, x, training):
        input_shape = tf.shape(x)
        return dropout(x, self.keep_prob, training, noise_shape=[input_shape[0], 1, input_shape[2]])


class ReduceSequence(Layer):
    def __init__(self, reduce="mean", name="reduce_sequence"):
        super(ReduceSequence, self).__init__(name)
        self.reduce = reduce

    def __call__(self, x, mask=None):
        if mask is not None:
            valid_mask = tf.expand_dims(
                tf.cast(tf.sequence_mask(mask, tf.shape(x)[1]), tf.float32), 2)

        if self.reduce == "max":
            if mask is not None:
                x += (1.0 - valid_mask) * tf.float32.min
            return tf.reduce_max(x, axis=1)
        elif self.reduce == "mean":
            if mask is not None:
                x *= valid_mask
                return tf.reduce_sum(x, axis=1) / (tf.cast(tf.expand_dims(mask, 1), tf.float32) + 1e-16)
            else:
                return tf.reduce_mean(x, axis=1)
        elif self.reduce == "sum":
            if valid_mask is not None:
                x *= valid_mask
            return tf.reduce_sum(x, axis=1)
        else:
            raise ValueError()


class Conv1DAndMaxPooling(Layer):
    """ Conv1D for 3D or 4D input tensor, the second-to-last dimension is regarded as timestep """

    def __init__(self, filters, kernel_size, strides=1, padding='valid', activation=tf.nn.relu,
                 name="conv1d_and_max_pooling"):
        super(Conv1DAndMaxPooling, self).__init__(name)
        self.conv_layer = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides,
                                                 padding=padding, activation=activation)

    def __call__(self, x, seq_len=None):
        input_shape = x.shape.as_list()
        batch_size = None
        if len(input_shape) == 4:
            batch_size = tf.shape(x)[0]
            seq_length = tf.shape(x)[1]
            x = tf.reshape(x, (-1, tf.shape(x)[-2], input_shape[-1]))
            x = self.conv_layer(x)
            if seq_len is not None:
                hidden_units = x.shape.as_list()[-1]
                x = tf.reshape(x, (batch_size, seq_length, tf.shape(x)[1], hidden_units))
                x = self.max_pooling(x, seq_len)
            else:
                x = tf.reduce_max(x, axis=1)
                x = tf.reshape(x, (batch_size, -1, x.shape.as_list()[-1]))
        elif len(input_shape) == 3:
            x = self.conv_layer(x)
            x = tf.reduce_max(x, axis=1)
        else:
            raise ValueError()

        return x

    def max_pooling(self, inputs, seq_len=None):
        rank = len(inputs.shape) - 2
        if seq_len is not None:
            shape = tf.shape(inputs)
            mask = tf.sequence_mask(tf.reshape(seq_len, (-1,)), shape[-2])
            mask = tf.cast(tf.reshape(mask, (shape[0], shape[1], shape[2], 1)), tf.float32)
            inputs = inputs * mask + (1 - mask) * VERY_NEGATIVE_NUMBER
        return tf.reduce_max(inputs, axis=rank)


class MultiConv1DAndMaxPooling(Layer):
    def __init__(self, filters, kernel_sizes, strides=1, padding='valid', activation=None,
                 name="multi_conv1d_and_max_pooling"):
        super(MultiConv1DAndMaxPooling, self).__init__(name)
        self.conv_layers = [Conv1DAndMaxPooling(filters, kernel_size, strides=strides,
                                                padding=padding, activation=activation,
                                                name="conv1d_and_max_pooling" + str(kernel_size))
                            for kernel_size in kernel_sizes]

    def __call__(self, x):
        return tf.concat([layer(x) for layer in self.conv_layers], axis=-1)


class MultiLayerRNN(Layer):
    def __init__(self, layers=None, concat_layer_out=True, input_keep_prob=1.0, name='multi_layer_rnn'):
        super(MultiLayerRNN, self).__init__(name)
        self.concat_layer_output = concat_layer_out
        self.dropout = VariationalDropout(input_keep_prob)
        self.rnn_layers = layers

    def __call__(self, x, x_len, training):
        output = x
        outputs = []
        for layer in self.rnn_layers:
            output, _ = layer(self.dropout(output, training), x_len)
            outputs.append(output)
        if self.concat_layer_output:
            return tf.concat(outputs, axis=-1)
        return outputs[-1]


class MultiHeadAttention(Layer):
    def __init__(self, heads, units, attention_on_itself=True, name='encoder_block'):
        super(MultiHeadAttention, self).__init__(name)
        self.heads = heads
        self.units = units
        self.attention_on_itself = attention_on_itself  # only workable when query==key
        self.dense_layers = [tf.keras.layers.Dense(units) for _ in range(3)]

    def __call__(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]
        max_query_len = tf.shape(query)[1]
        max_key_len = tf.shape(key)[1]
        wq = tf.transpose(
            tf.reshape(self.dense_layers[0](query), [batch_size, max_query_len, self.heads, self.units // self.heads]),
            [2, 0, 1, 3])  # Head*B*QL*(U/Head)
        wk = tf.transpose(
            tf.reshape(self.dense_layers[1](key), [batch_size, max_key_len, self.heads, self.units // self.heads]),
            [2, 0, 1, 3])  # Head*B*KL*(U/Head)
        wv = tf.transpose(
            tf.reshape(self.dense_layers[2](value), [batch_size, max_key_len, self.heads, self.units // self.heads]),
            [2, 0, 1, 3])  # Head*B*KL*(U/Head)
        attention_score = tf.matmul(wq, wk, transpose_b=True) / tf.sqrt(float(self.units) / self.heads)  # Head*B*QL*KL
        if query == key and not self.attention_on_itself:
            attention_score += tf.matrix_diag(tf.zeros(max_key_len) - 100.0)
        if mask is not None:
            attention_score += tf.expand_dims(mask, 1)
        similarity = tf.nn.softmax(attention_score, -1)  # Head*B*QL*KL
        return tf.reshape(tf.transpose(tf.matmul(similarity, wv), [1, 2, 0, 3]),
                          [batch_size, max_query_len, self.units])  # B*QL*U


class EncoderBlock(Layer):
    def __init__(self, kernel_size, filters, conv_layers, heads, keep_prob=1.0, name='encoder_block'):
        super(EncoderBlock, self).__init__(name)
        self.kernel_size = kernel_size
        self.filters = filters
        self.convs = [tf.keras.layers.SeparableConv1D(filters, kernel_size, padding='same', activation=tf.nn.relu) for _
                      in range(conv_layers)]
        self.dense1 = tf.keras.layers.Dense(filters, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(filters)
        self.keep_prob = keep_prob
        self.multihead_attention = MultiHeadAttention(heads, filters)
        self.dropout = Dropout(self.keep_prob)

    def __call__(self, x, training, mask=None):
        for conv in self.convs:
            norm_x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
            x += self.dropout(conv(norm_x), training)
        norm_x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        x += self.dropout(self.multihead_attention(norm_x, norm_x, norm_x, mask), training)
        norm_x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        x += self.dropout(self.dense2(self.dropout(self.dense1(norm_x), training)), training)
        return x


class ElmoEmbedding(Layer):
    def __init__(self, elmo_url='https://tfhub.dev/google/elmo/2', local_path=None,
                 trainable=True, name='elmo_embedding'):
        super(ElmoEmbedding, self).__init__(name)
        self.module_path = elmo_url if local_path is None else local_path
        self.elmo = hub.Module(self.module_path, trainable=trainable)

    def __call__(self, tokens_input, tokens_length):
        embedding = self.elmo(inputs={"tokens": tokens_input, 'sequence_len': tokens_length},
                              signature="tokens",
                              as_dict=True)["elmo"]
        return embedding


class PartiallyTrainableEmbedding(Layer):
    """
    Special embedding layer of which the top K embeddings are trainable and the rest is fixed.
    A technique used in DrQA and FusionNet.
    Note that the trainable K embeddings are the first K rows of the embedding matrix for convenience.
    """

    def __init__(self, trainable_num=1000, pretrained_embedding=None, embedding_shape=None, init_scale=0.02,
                 name="partially_trainable_embedding"):
        # If pretrained embedding is None, embedding_shape must be specified and
        # embedding matrix will be randomly initialized.
        super(PartiallyTrainableEmbedding, self).__init__(name)
        if pretrained_embedding is None and embedding_shape is None:
            raise ValueError("At least one of pretrained_embedding and embedding_shape must be specified!")

        input_shape = pretrained_embedding.shape if pretrained_embedding is not None else embedding_shape
        if not (0 < trainable_num < input_shape[0]):
            raise ValueError("trainable_num must be greater that 0 and less than vocabulary size!")

        with tf.variable_scope(self.name):
            trainable_embedding_init = tf.constant_initializer(pretrained_embedding[:trainable_num, :]) \
                if pretrained_embedding is not None else tf.random_uniform_initializer(-init_scale, init_scale)
            fixed_embedding_init = tf.constant_initializer(pretrained_embedding[trainable_num:, :]) \
                if pretrained_embedding is not None else tf.random_uniform_initializer(-init_scale, init_scale)
            trainable_embedding = tf.get_variable('trainable_embedding', shape=(trainable_num, input_shape[1]),
                                                  initializer=trainable_embedding_init,
                                                  trainable=True)
            fixed_embeding = tf.get_variable('fix_embedding', shape=(input_shape[0] - trainable_num, input_shape[1]),
                                             initializer=fixed_embedding_init, trainable=False)
            self.embedding = tf.concat([trainable_embedding, fixed_embeding], axis=0)

    def __call__(self, indices):
        return tf.nn.embedding_lookup(self.embedding, indices)


class Embedding(Layer):
    def __init__(self, pretrained_embedding=None, embedding_shape=None, trainable=True, init_scale=0.02,
                 name="embedding"):
        super(Embedding, self).__init__(name)
        if pretrained_embedding is None and embedding_shape is None:
            raise ValueError("At least one of pretrained_embedding and embedding_shape must be specified!")
        input_shape = pretrained_embedding.shape if pretrained_embedding is not None else embedding_shape

        with tf.variable_scope(self.name):
            embedding_init = tf.constant_initializer(pretrained_embedding) \
                if pretrained_embedding is not None else tf.random_uniform_initializer(-init_scale, init_scale)
            self.embedding = tf.get_variable('embedding', shape=input_shape,
                                             initializer=embedding_init, trainable=trainable)

    def __call__(self, indices):
        return tf.nn.embedding_lookup(self.embedding, indices)


class CoveEmbedding(Layer):
    def __init__(self, cove_path, pretrained_word_embedding=None, vocab=None, word_embedding_size=300,
                 name='cove_embedding'):
        super(CoveEmbedding, self).__init__(name)
        if pretrained_word_embedding is None:
            raise ValueError("pretrained glove word embedding must be specified ! ")
        self.word_embedding_for_cove = Embedding(pretrained_word_embedding,
                                                 embedding_shape=(len(vocab.get_word_vocab()) + 1, word_embedding_size),
                                                 trainable=False)
        self.cove_model = tf.keras.models.load_model(cove_path)
        self.cove_model.trainable = False

    def __call__(self, input, input_len):
        word_embedding_repr = self.word_embedding_for_cove(input)
        return tf.stop_gradient(self.cove_model(word_embedding_repr, input_len))


class SumMaxEncoder(Layer):
    def __init__(self, name="sum_max_encoder"):
        super(SumMaxEncoder, self).__init__(name)

    def __call__(self, x, seq_len, max_len=None):
        mask_x1 = add_seq_mask(x, seq_len, 'mul', max_len)
        mask_x2 = add_seq_mask(x, seq_len, 'add', max_len)
        a = tf.reduce_sum(mask_x1, 1)
        b = tf.reduce_max(mask_x2, 1)
        ret = tf.concat([a, b], axis=1)
        ret = tf.expand_dims(ret, 1)
        return ret


class BertEmbedding(Layer):
    def __init__(self, BERT_PRETRAINED_DIR='/uncased_L-12_H-768_A-12/', name='bert_model_helper'):
        super(BertEmbedding, self).__init__(name)
        CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
        self.bert_config = modeling.BertConfig.from_json_file(CONFIG_FILE)
        self.init_checkpoint = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

    def __call__(self, input_ids, input_mask, segment_ids, is_training,use_one_hot_embeddings=True,return_pool_output=False):
        """Creates a classification model."""
        self.model = modeling.BertModel(
                config=self.bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings)
        return self.model.get_sequence_output() if not return_pool_output else  (self.model.get_sequence_output(),self.model.get_pooled_output())

    def init_bert(self):
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        init_checkpoint = self.init_checkpoint
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
