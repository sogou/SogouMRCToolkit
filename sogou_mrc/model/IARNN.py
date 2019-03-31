# coding:utf-8
import os
import tensorflow as tf
import numpy as np
import logging
from collections import OrderedDict, defaultdict
from sogou_mrc.model.base_model import BaseModel
from sogou_mrc.nn.layers import Conv1DAndMaxPooling, Dropout, Highway, Embedding
from sogou_mrc.nn.recurrent import CudnnBiLSTM
from sogou_mrc.nn.attention import BiAttention
from sogou_mrc.nn.similarity_function import TriLinear, BiLinear, ProjectedDotProduct
from sogou_mrc.nn.ops import masked_softmax, weighted_sum, mask_logits


class IARNN_word(BaseModel):
    def __init__(self, vocab, pretrained_word_embedding=None, word_embedding_size=100, char_embedding_size=8,
                 char_conv_filters=100,
                 char_conv_kernel_size=5, rnn_hidden_size=100,
                 dropout_keep_prob=0.8, max_answer_len=17, word_embedding_trainable=False):
        super(IARNN_word, self).__init__(vocab)
        self.rnn_hidden_size = rnn_hidden_size
        self.keep_prob = dropout_keep_prob
        self.word_embedding_size = word_embedding_size
        self.pretrained_word_embedding = pretrained_word_embedding
        self.char_embedding_size = char_embedding_size
        self.char_conv_filters = char_conv_filters
        self.char_conv_kernel_size = char_conv_kernel_size
        self.max_answer_len = max_answer_len
        self.word_embedding_trainable = word_embedding_trainable
        self._build_graph()

    def _build_graph(self):
        self.context_word = tf.placeholder(tf.int32, [None, None])
        self.context_char = tf.placeholder(tf.int32, [None, None, None])
        self.context_len = tf.placeholder(tf.int32, [None])
        self.question_word = tf.placeholder(tf.int32, [None, None])
        self.question_char = tf.placeholder(tf.int32, [None, None, None])
        self.question_len = tf.placeholder(tf.int32, [None])
        self.answer_start = tf.placeholder(tf.int32, [None])
        self.answer_end = tf.placeholder(tf.int32, [None])
        self.training = tf.placeholder(tf.bool, [])

        # 1. Word encoding
        word_embedding = Embedding(pretrained_embedding=self.pretrained_word_embedding,
                                   embedding_shape=(len(self.vocab.get_word_vocab()) + 1, self.word_embedding_size),
                                   trainable=self.word_embedding_trainable)

        context_word_repr = word_embedding(self.context_word)
        question_word_repr = word_embedding(self.question_word)

        dropout = Dropout(self.keep_prob)

        # 2 inner attention between question word and context word

        inner_att = ProjectedDotProduct(self.rnn_hidden_size, activation=tf.nn.leaky_relu, reuse_weight=True)

        inner_score = inner_att(question_word_repr, context_word_repr)

        context_word_softmax = tf.nn.softmax(inner_score, axis=2)

        question_inner_representation = tf.matmul(context_word_softmax, context_word_repr)

        question_word_softmax = tf.nn.softmax(inner_score, axis=1)

        context_inner_representation = tf.matmul(question_word_softmax, question_word_repr, transpose_a=True)

        highway1 = Highway()
        highway2 = Highway()

        context_repr = highway1(highway2(tf.concat([context_word_repr, context_inner_representation], axis=-1)))
        question_repr = highway1(highway2(tf.concat([question_word_repr, question_inner_representation], axis=-1)))

        # 2. Phrase encoding
        phrase_lstm = CudnnBiLSTM(self.rnn_hidden_size)
        context_repr, _ = phrase_lstm(dropout(context_repr, self.training), self.context_len)
        question_repr, _ = phrase_lstm(dropout(question_repr, self.training), self.question_len)

        # 3. Bi-Attention
        bi_attention = BiAttention(TriLinear())
        c2q, q2c = bi_attention(context_repr, question_repr, self.context_len, self.question_len)

        # 4. Modeling layer
        final_merged_context = tf.concat([context_repr, c2q, context_repr * c2q, context_repr * q2c], axis=-1)
        modeling_lstm1 = CudnnBiLSTM(self.rnn_hidden_size)
        modeling_lstm2 = CudnnBiLSTM(self.rnn_hidden_size)
        modeled_context1, _ = modeling_lstm1(dropout(final_merged_context, self.training), self.context_len)
        modeled_context2, _ = modeling_lstm2(dropout(modeled_context1, self.training), self.context_len)
        modeled_context = modeled_context1 + modeled_context2

        # 5. Start prediction
        start_pred_layer = tf.keras.layers.Dense(1, use_bias=False)
        start_logits = start_pred_layer(
            dropout(tf.concat([final_merged_context, modeled_context], axis=-1), self.training))
        start_logits = tf.squeeze(start_logits, axis=-1)
        self.start_prob = masked_softmax(start_logits, self.context_len)

        # 6. End prediction
        start_repr = weighted_sum(modeled_context, self.start_prob)
        tiled_start_repr = tf.tile(tf.expand_dims(start_repr, axis=1), [1, tf.shape(modeled_context)[1], 1])
        end_lstm = CudnnBiLSTM(self.rnn_hidden_size)
        encoded_end_repr, _ = end_lstm(dropout(tf.concat(
            [final_merged_context, modeled_context, tiled_start_repr, modeled_context * tiled_start_repr], axis=-1),
            self.training),
            self.context_len)
        end_pred_layer = tf.keras.layers.Dense(1, use_bias=False)
        end_logits = end_pred_layer(dropout(tf.concat(
            [final_merged_context, encoded_end_repr], axis=-1), self.training))
        end_logits = tf.squeeze(end_logits, axis=-1)
        self.end_prob = masked_softmax(end_logits, self.context_len)

        # 7. Loss and input/output dict
        self.start_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mask_logits(start_logits, self.context_len),
                                                           labels=self.answer_start))
        self.end_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mask_logits(end_logits, self.context_len),
                                                           labels=self.answer_end))
        self.loss = (self.start_loss + self.end_loss) / 2
        global_step = tf.train.get_or_create_global_step()

        self.input_placeholder_dict = OrderedDict({
            "context_word": self.context_word,
            "context_char": self.context_char,
            "context_len": self.context_len,
            "question_word": self.question_word,
            "question_char": self.question_char,
            "question_len": self.question_len,
            "answer_start": self.answer_start,
            "answer_end": self.answer_end,
            "training": self.training
        })

        self.output_variable_dict = OrderedDict({
            "start_prob": self.start_prob,
            "end_prob": self.end_prob
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

    def train_and_evaluate(self, train_instances, eval_instances, batch_size, evaluator, epochs=1, eposides=1,
                           save_dir=None, summary_dir=None, save_summary_steps=10):
        if not self.initialized:
            self.session.run(tf.global_variables_initializer())

        train_input, train_iterator_init_op = BaseModel._make_input(self.vocab, train_instances, batch_size,
                                                                    training=True)
        eval_input, eval_iterator_init_op = BaseModel._make_input(self.vocab, eval_instances, batch_size,
                                                                  training=False)
        self.session.run(tf.tables_initializer())

        IARNN_word._train_and_evaluate(self, train_instances, train_input, train_iterator_init_op,
                                       eval_instances, eval_input, eval_iterator_init_op,
                                       batch_size, evaluator, epochs=epochs, eposides=eposides,
                                       save_dir=save_dir, summary_dir=summary_dir,
                                       save_summary_steps=save_summary_steps)

    def evaluate(self, eval_instances, batch_size, evaluator):
        if not self.initialized:
            self.session.run(tf.global_variables_initializer())

        eval_input, eval_iterator_init_op = BaseModel._make_input(self.vocab, eval_instances, batch_size,
                                                                  training=False)
        self.session.run(tf.tables_initializer())
        IARNN_word._evaluate(self, eval_instances, eval_input, eval_iterator_init_op, batch_size, evaluator)

    def get_best_answer(self, output, instances):
        answer_list = []
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
            answer_list.append(pred_answer)
        return answer_list


class IARNN_hidden(BaseModel):
    def __init__(self, vocab, pretrained_word_embedding=None, word_embedding_size=100, char_embedding_size=8,
                 char_conv_filters=100,
                 char_conv_kernel_size=5, rnn_hidden_size=100,
                 dropout_keep_prob=0.8, max_answer_len=17, word_embedding_trainable=False):
        super(IARNN_hidden, self).__init__(vocab)
        self.rnn_hidden_size = rnn_hidden_size
        self.keep_prob = dropout_keep_prob
        self.word_embedding_size = word_embedding_size
        self.pretrained_word_embedding = pretrained_word_embedding
        self.char_embedding_size = char_embedding_size
        self.char_conv_filters = char_conv_filters
        self.char_conv_kernel_size = char_conv_kernel_size
        self.max_answer_len = max_answer_len
        self.word_embedding_trainable = word_embedding_trainable
        self._build_graph()

    def _build_graph(self):
        self.context_word = tf.placeholder(tf.int32, [None, None])
        self.context_char = tf.placeholder(tf.int32, [None, None, None])
        self.context_len = tf.placeholder(tf.int32, [None])
        self.question_word = tf.placeholder(tf.int32, [None, None])
        self.question_char = tf.placeholder(tf.int32, [None, None, None])
        self.question_len = tf.placeholder(tf.int32, [None])
        self.answer_start = tf.placeholder(tf.int32, [None])
        self.answer_end = tf.placeholder(tf.int32, [None])
        self.training = tf.placeholder(tf.bool, [])

        # 1. Word encoding
        word_embedding = Embedding(pretrained_embedding=self.pretrained_word_embedding,
                                   embedding_shape=(len(self.vocab.get_word_vocab()) + 1, self.word_embedding_size),
                                   trainable=self.word_embedding_trainable)

        context_word_repr = word_embedding(self.context_word)
        question_word_repr = word_embedding(self.question_word)

        dropout = Dropout(self.keep_prob)

        # 2. Phrase encoding
        question_lstm = CudnnBiLSTM(self.rnn_hidden_size)
        context_lstm = CudnnBiLSTM(self.rnn_hidden_size)

        question_repr, _ = question_lstm(dropout(question_word_repr, self.training), self.question_len)

        inner_att = ProjectedDotProduct(self.rnn_hidden_size, activation=tf.nn.leaky_relu, reuse_weight=False)

        inner_score = inner_att(context_word_repr, question_repr)

        question_word_softmax = tf.nn.softmax(inner_score, axis=2)

        context_inner_representation = tf.matmul(question_word_softmax, question_repr)

        highway1 = Highway()
        highway2 = Highway()

        context_repr = highway1(highway2(tf.concat([context_word_repr, context_inner_representation], axis=-1)))

        context_repr, _ = context_repr(dropout(context_repr, self.training), self.context_len)

        # 3. Bi-Attention
        bi_attention = BiAttention(TriLinear())
        c2q, q2c = bi_attention(context_repr, question_repr, self.context_len, self.question_len)

        # 4. Modeling layer
        final_merged_context = tf.concat([context_repr, c2q, context_repr * c2q, context_repr * q2c], axis=-1)
        modeling_lstm1 = CudnnBiLSTM(self.rnn_hidden_size)
        modeling_lstm2 = CudnnBiLSTM(self.rnn_hidden_size)
        modeled_context1, _ = modeling_lstm1(dropout(final_merged_context, self.training), self.context_len)
        modeled_context2, _ = modeling_lstm2(dropout(modeled_context1, self.training), self.context_len)
        modeled_context = modeled_context1 + modeled_context2

        # 5. Start prediction
        start_pred_layer = tf.keras.layers.Dense(1, use_bias=False)
        start_logits = start_pred_layer(
            dropout(tf.concat([final_merged_context, modeled_context], axis=-1), self.training))
        start_logits = tf.squeeze(start_logits, axis=-1)
        self.start_prob = masked_softmax(start_logits, self.context_len)

        # 6. End prediction
        start_repr = weighted_sum(modeled_context, self.start_prob)
        tiled_start_repr = tf.tile(tf.expand_dims(start_repr, axis=1), [1, tf.shape(modeled_context)[1], 1])
        end_lstm = CudnnBiLSTM(self.rnn_hidden_size)
        encoded_end_repr, _ = end_lstm(dropout(tf.concat(
            [final_merged_context, modeled_context, tiled_start_repr, modeled_context * tiled_start_repr], axis=-1),
            self.training),
            self.context_len)
        end_pred_layer = tf.keras.layers.Dense(1, use_bias=False)
        end_logits = end_pred_layer(dropout(tf.concat(
            [final_merged_context, encoded_end_repr], axis=-1), self.training))
        end_logits = tf.squeeze(end_logits, axis=-1)
        self.end_prob = masked_softmax(end_logits, self.context_len)

        # 7. Loss and input/output dict
        self.start_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mask_logits(start_logits, self.context_len),
                                                           labels=self.answer_start))
        self.end_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mask_logits(end_logits, self.context_len),
                                                           labels=self.answer_end))
        self.loss = (self.start_loss + self.end_loss) / 2
        global_step = tf.train.get_or_create_global_step()

        self.input_placeholder_dict = OrderedDict({
            "context_word": self.context_word,
            "context_char": self.context_char,
            "context_len": self.context_len,
            "question_word": self.question_word,
            "question_char": self.question_char,
            "question_len": self.question_len,
            "answer_start": self.answer_start,
            "answer_end": self.answer_end,
            "training": self.training
        })

        self.output_variable_dict = OrderedDict({
            "start_prob": self.start_prob,
            "end_prob": self.end_prob
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

    def train_and_evaluate(self, train_instances, eval_instances, batch_size, evaluator, epochs=1, eposides=1,
                           save_dir=None, summary_dir=None, save_summary_steps=10):
        if not self.initialized:
            self.session.run(tf.global_variables_initializer())

        train_input, train_iterator_init_op = BaseModel._make_input(self.vocab, train_instances, batch_size,
                                                                    training=True)
        eval_input, eval_iterator_init_op = BaseModel._make_input(self.vocab, eval_instances, batch_size,
                                                                  training=False)
        self.session.run(tf.tables_initializer())

        IARNN_hidden._train_and_evaluate(self, train_instances, train_input, train_iterator_init_op,
                                         eval_instances, eval_input, eval_iterator_init_op,
                                         batch_size, evaluator, epochs=epochs, eposides=eposides,
                                         save_dir=save_dir, summary_dir=summary_dir,
                                         save_summary_steps=save_summary_steps)

    def evaluate(self, eval_instances, batch_size, evaluator):
        if not self.initialized:
            self.session.run(tf.global_variables_initializer())

        eval_input, eval_iterator_init_op = BaseModel._make_input(self.vocab, eval_instances, batch_size,
                                                                  training=False)
        self.session.run(tf.tables_initializer())
        IARNN_hidden._evaluate(self, eval_instances, eval_input, eval_iterator_init_op, batch_size, evaluator)

    def get_best_answer(self, output, instances):
        answer_list = []
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
            answer_list.append(pred_answer)
        return answer_list
