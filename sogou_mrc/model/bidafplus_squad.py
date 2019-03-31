# coding:utf-8
import os
import tensorflow as tf
import numpy as np
from collections import OrderedDict, defaultdict
from sogou_mrc.model.base_model import BaseModel
from sogou_mrc.nn.layers import Conv1DAndMaxPooling, Dropout, VariationalDropout, Highway, Embedding, SumMaxEncoder
from sogou_mrc.nn.recurrent import CudnnBiGRU
from sogou_mrc.nn.attention import BiAttention, SelfAttention
from sogou_mrc.nn.similarity_function import TriLinear
from sogou_mrc.nn.ops import masked_softmax, weighted_sum, mask_logits

VERY_NEGATIVE_NUMBER = -1e29

class BiDAFPlusSQuad(BaseModel):
    def __init__(self, vocab, pretrained_word_embedding=None, word_embedding_size=100, char_embedding_size=20, char_conv_filters=100,
                 char_conv_kernel_size=5, rnn_hidden_size=100,
                 dropout_keep_prob=0.8, max_answer_len=11, word_embedding_trainable=False, ema_decay=0.):
        super(BiDAFPlusSQuad, self).__init__(vocab)
        self.rnn_hidden_size = rnn_hidden_size
        self.keep_prob = dropout_keep_prob
        self.word_embedding_size = word_embedding_size
        self.pretrained_word_embedding = pretrained_word_embedding
        self.char_embedding_size = char_embedding_size
        self.char_conv_filters = char_conv_filters
        self.char_conv_kernel_size = char_conv_kernel_size
        self.max_answer_len = max_answer_len
        self.word_embedding_trainable = word_embedding_trainable
        self.ema_decay = ema_decay
        self._build_graph()

    def _build_graph(self):
        self.context_word = tf.placeholder(tf.int32, [None, None])
        self.context_len = tf.placeholder(tf.int32, [None])

        self.context_char = tf.placeholder(tf.int32, [None, None, None])
        self.context_word_len = tf.placeholder(tf.int32, [None, None])

        self.question_word = tf.placeholder(tf.int32, [None, None])
        self.question_len = tf.placeholder(tf.int32, [None])

        self.question_char = tf.placeholder(tf.int32, [None, None, None])
        self.question_word_len = tf.placeholder(tf.int32, [None, None])

        self.answer_start = tf.placeholder(tf.int32, [None])
        self.answer_end = tf.placeholder(tf.int32, [None])
        self.training = tf.placeholder(tf.bool, [])

        # 1. Word encoding
        word_embedding = Embedding(pretrained_embedding=self.pretrained_word_embedding,
                                   embedding_shape=(len(self.vocab.get_word_vocab()) + 1, self.word_embedding_size),
                                   trainable=self.word_embedding_trainable)
        char_embedding = Embedding(embedding_shape=(len(self.vocab.get_char_vocab()) + 1, self.char_embedding_size), trainable=True, init_scale=0.05)

        # 1.1 Embedding
        context_word_repr = word_embedding(self.context_word)
        context_char_repr = char_embedding(self.context_char)
        question_word_repr = word_embedding(self.question_word)
        question_char_repr = char_embedding(self.question_char)

        self.context_word_repr = context_word_repr
        self.question_word_repr = question_word_repr
        self.context_char_repr = context_char_repr
        self.question_char_repr = question_char_repr

        # 1.2 Char convolution
        dropout = Dropout(self.keep_prob)
        conv1d = Conv1DAndMaxPooling(self.char_conv_filters, self.char_conv_kernel_size)
        question_char_repr = conv1d(dropout(question_char_repr, self.training), self.question_word_len)
        context_char_repr = conv1d(dropout(context_char_repr, self.training), self.context_word_len)

        # 2. Phrase encoding
        context_repr = tf.concat([context_word_repr, context_char_repr], axis=-1)
        question_repr = tf.concat([question_word_repr, question_char_repr], axis=-1)

        variational_dropout = VariationalDropout(self.keep_prob)
        emb_enc_gru = CudnnBiGRU(self.rnn_hidden_size)

        context_repr = variational_dropout(context_repr, self.training)
        context_repr, _ = emb_enc_gru(context_repr, self.context_len)
        context_repr = variational_dropout(context_repr,  self.training)

        question_repr = variational_dropout(question_repr, self.training)
        question_repr, _ = emb_enc_gru(question_repr, self.question_len)
        question_repr = variational_dropout(question_repr, self.training)

        # 3. Bi-Attention
        bi_attention = BiAttention(TriLinear(bias=True, name="bi_attention_tri_linear"))
        c2q, q2c = bi_attention(context_repr, question_repr, self.context_len, self.question_len)
        context_repr = tf.concat([context_repr, c2q, context_repr * c2q, context_repr * q2c], axis=-1)

        # 4. Self-Attention layer
        dense1 = tf.keras.layers.Dense(self.rnn_hidden_size*2, use_bias=True, activation=tf.nn.relu)
        gru = CudnnBiGRU(self.rnn_hidden_size)
        dense2 = tf.keras.layers.Dense(self.rnn_hidden_size*2, use_bias=True, activation=tf.nn.relu)
        self_attention = SelfAttention(TriLinear(bias=True, name="self_attention_tri_linear"))

        inputs = dense1(context_repr)
        outputs = variational_dropout(inputs, self.training)
        outputs, _ = gru(outputs, self.context_len)
        outputs = variational_dropout(outputs, self.training)
        c2c = self_attention(outputs, self.context_len)
        outputs = tf.concat([c2c, outputs, c2c * outputs], axis=len(c2c.shape)-1)
        outputs = dense2(outputs)
        context_repr = inputs + outputs
        context_repr = variational_dropout(context_repr, self.training)
        
        # 5. Modeling layer
        sum_max_encoding = SumMaxEncoder()
        context_modeling_gru1 = CudnnBiGRU(self.rnn_hidden_size)
        context_modeling_gru2 = CudnnBiGRU(self.rnn_hidden_size)
        question_modeling_gru1 = CudnnBiGRU(self.rnn_hidden_size)
        question_modeling_gru2 = CudnnBiGRU(self.rnn_hidden_size)
        self.max_context_len = tf.reduce_max(self.context_len)
        self.max_question_len = tf.reduce_max(self.question_len)

        modeled_context1, _ = context_modeling_gru1(context_repr, self.context_len)
        modeled_context2, _ = context_modeling_gru2(tf.concat([context_repr, modeled_context1], axis=2), self.context_len)
        encoded_context = sum_max_encoding(modeled_context1, self.context_len, self.max_context_len)
        modeled_question1, _ = question_modeling_gru1(question_repr, self.question_len)
        modeled_question2, _ = question_modeling_gru2(tf.concat([question_repr, modeled_question1], axis=2), self.question_len)
        encoded_question = sum_max_encoding(modeled_question2, self.question_len, self.max_question_len)

        start_dense = tf.keras.layers.Dense(1, activation=None)
        start_logits = tf.squeeze(start_dense(modeled_context1), squeeze_dims=[2])
        start_logits = mask_logits(start_logits, self.context_len)
        start_prob = tf.nn.softmax(start_logits)

        end_dense = tf.keras.layers.Dense(1, activation=None)
        end_logits = tf.squeeze(end_dense(modeled_context2), squeeze_dims=[2])
        end_logits = mask_logits(end_logits, self.context_len)
        end_prob = tf.nn.softmax(end_logits)

        self.start_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=start_logits,
                                                           labels=self.answer_start))
        self.end_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_logits,
                                                           labels=self.answer_end))
        # 7. Loss and input/output dict

        self.loss = self.start_loss + self.end_loss
        global_step = tf.train.get_or_create_global_step()

        self.input_placeholder_dict = OrderedDict({
            "context_word": self.context_word,
            "context_len": self.context_len,
            "context_char": self.context_char,
            "context_word_len": self.context_word_len,
            "question_word": self.question_word,
            "question_len": self.question_len,
            "question_char": self.question_char,
            "question_word_len": self.question_word_len,
            "answer_start": self.answer_start,
            "answer_end": self.answer_end,
            "training": self.training
        })

        self.output_variable_dict = OrderedDict({
            "start_prob": start_prob,
            "end_prob": end_prob,
        })

        # 8. Metrics and summary
        with tf.variable_scope("train_metrics"):
            self.train_metrics = {
                'loss': tf.metrics.mean(self.loss)
            }

        self.train_update_metrics = tf.group(*[op for _, op in self.train_metrics.values()])
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="train_metrics")
        self.train_metric_init_op = tf.variables_initializer(metric_variables)

        with tf.variable_scope("eval_metrics"):
            self.eval_metrics = {
                'loss': tf.metrics.mean(self.loss)
            }

        self.eval_update_metrics = tf.group(*[op for _, op in self.eval_metrics.values()])
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="eval_metrics")
        self.eval_metric_init_op = tf.variables_initializer(metric_variables)

        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

    def compile(self, optimizer, initial_lr):
        self.optimizer = optimizer(initial_lr)
        self.train_op = self.optimizer.minimize(self.loss)
    
    def get_best_answer(self, output, instances):
        preds_dict = {}
        for i in range(len(instances)):
            instance = instances[i]
            max_prob, max_start, max_end = 0, 0, 0
            for end in range(output['end_prob'][i].shape[0]):
                for start in range(max(0, end - self.max_answer_len + 1), end + 1):
                    prob = output["start_prob"][i][start] * output["end_prob"][i][end]
                    if prob > max_prob:
                        max_start, max_end = start, end
                        max_prob = prob
            char_start_position = instance["context_token_spans"][max_start][0]
            char_end_position = instance["context_token_spans"][max_end][1]
            pred_answer = instance["context"][char_start_position:char_end_position]
            preds_dict[instance['qid']] = pred_answer
        return preds_dict
