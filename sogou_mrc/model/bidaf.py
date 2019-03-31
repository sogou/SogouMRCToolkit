# coding:utf-8
import os
import tensorflow as tf
import numpy as np
import logging
from collections import OrderedDict, defaultdict
from sogou_mrc.model.base_model import BaseModel
from sogou_mrc.nn.layers import Conv1DAndMaxPooling, Dropout, Highway, Embedding,ElmoEmbedding
from sogou_mrc.nn.recurrent import CudnnBiLSTM
from sogou_mrc.nn.attention import BiAttention
from sogou_mrc.nn.similarity_function import TriLinear
from sogou_mrc.nn.ops import masked_softmax, weighted_sum, mask_logits


class BiDAF(BaseModel):
    def __init__(self, vocab, pretrained_word_embedding=None, word_embedding_size=100, char_embedding_size=8,
                 char_conv_filters=100,
                 char_conv_kernel_size=5, rnn_hidden_size=100,
                 dropout_keep_prob=0.8, max_answer_len=17, word_embedding_trainable=False,use_elmo=False,elmo_local_path=None,
                 enable_na_answer=False):
        super(BiDAF, self).__init__(vocab)
        self.rnn_hidden_size = rnn_hidden_size
        self.keep_prob = dropout_keep_prob
        self.word_embedding_size = word_embedding_size
        self.pretrained_word_embedding = pretrained_word_embedding
        self.char_embedding_size = char_embedding_size
        self.char_conv_filters = char_conv_filters
        self.char_conv_kernel_size = char_conv_kernel_size
        self.max_answer_len = max_answer_len
        self.use_elmo = use_elmo
        self.elmo_local_path= elmo_local_path
        self.word_embedding_trainable = word_embedding_trainable
        self.enable_na_answer = enable_na_answer # for squad2.0
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

        self.question_tokens = tf.placeholder(tf.string, [None, None])
        self.context_tokens = tf.placeholder(tf.string,[None,None])
        if self.enable_na_answer:
            self.na = tf.placeholder(tf.int32, [None])
        # 1. Word encoding
        word_embedding = Embedding(pretrained_embedding=self.pretrained_word_embedding,
                                   embedding_shape=(len(self.vocab.get_word_vocab()) + 1, self.word_embedding_size),
                                   trainable=self.word_embedding_trainable)
        char_embedding = Embedding(embedding_shape=(len(self.vocab.get_char_vocab()) + 1, self.char_embedding_size),
                                   trainable=True, init_scale=0.2)


        # 1.1 Embedding
        context_word_repr = word_embedding(self.context_word)
        context_char_repr = char_embedding(self.context_char)
        question_word_repr = word_embedding(self.question_word)
        question_char_repr = char_embedding(self.question_char)

        # 1.2 Char convolution
        dropout = Dropout(self.keep_prob)
        conv1d = Conv1DAndMaxPooling(self.char_conv_filters, self.char_conv_kernel_size)
        context_char_repr = dropout(conv1d(context_char_repr), self.training)
        question_char_repr = dropout(conv1d(question_char_repr), self.training)

        #elmo embedding
        if self.use_elmo:
            elmo_emb = ElmoEmbedding(local_path=self.elmo_local_path)
            context_elmo_repr = elmo_emb(self.context_tokens,self.context_len)
            context_elmo_repr = dropout(context_elmo_repr,self.training)
            question_elmo_repr = elmo_emb(self.question_tokens,self.question_len)
            question_elmo_repr = dropout(question_elmo_repr,self.training)
        #concat word and char
        context_repr = tf.concat([context_word_repr, context_char_repr],axis=-1)
        question_repr = tf.concat([question_word_repr,question_char_repr],axis=-1)
        if self.use_elmo:
            context_repr= tf.concat([context_repr,context_elmo_repr],axis=-1)
            question_repr = tf.concat([question_repr,question_elmo_repr],axis=-1)

        # 1.3 Highway network
        highway1 = Highway()
        highway2 = Highway()
        context_repr = highway2(highway1(context_repr))
        question_repr = highway2(highway1(question_repr))

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
        if self.enable_na_answer:
            self.na_bias = tf.get_variable("na_bias", shape=[1], dtype='float')
            self.na_bias_tiled = tf.tile(tf.reshape(self.na_bias, [1, 1]), [tf.shape(self.context_word)[0], 1])
            self.concat_start_na_logits = tf.concat([self.na_bias_tiled, start_logits], axis=-1)
            concat_start_na_prob = masked_softmax(self.concat_start_na_logits, self.context_len + 1)
            self.na_prob = tf.squeeze(tf.slice(concat_start_na_prob, [0, 0], [-1, 1]), axis=1)
            self.start_prob = tf.slice(concat_start_na_prob, [0, 1], [-1, -1])
            self.concat_end_na_logits = tf.concat([self.na_bias_tiled,end_logits],axis=-1)
            concat_end_na_prob = masked_softmax(self.concat_end_na_logits,self.context_len+1)
            self.na_prob2 =tf.squeeze(tf.slice(concat_end_na_prob,[0,0],[-1,1]),axis=1)
            self.end_prob = tf.slice(concat_end_na_prob,[0,1],[-1,-1])
            max_len =tf.reduce_max(self.context_len)
            start_label = tf.cast(tf.one_hot(self.answer_start,max_len),tf.float32)
            start_label =(1.0-tf.cast(tf.expand_dims(self.na,axis=-1),tf.float32))*start_label
            na = tf.cast(tf.expand_dims(self.na,axis=-1),tf.float32)
            start_na_label = tf.concat([na,start_label],axis=-1)
            self.start_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=mask_logits(self.concat_start_na_logits,self.context_len+1),
                    labels=start_na_label))
            end_label = tf.cast(tf.one_hot(self.answer_end,max_len),tf.float32)
            end_label = (1.0-tf.cast(tf.expand_dims(self.na,axis=-1),tf.float32))*end_label
            end_na_label = tf.concat([na,end_label],axis=-1)
            self.end_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=mask_logits(self.concat_end_na_logits,self.context_len+1),
                    labels=end_na_label))
        else:

            self.start_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mask_logits(start_logits, self.context_len),
                                                               labels=self.answer_start))
            self.end_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mask_logits(end_logits, self.context_len),
                                                               labels=self.answer_end))
        self.loss = self.start_loss + self.end_loss
        global_step = tf.train.get_or_create_global_step()
        input_dict = {
            "context_word": self.context_word,
            "context_char": self.context_char,
            "context_len": self.context_len,
            "question_word": self.question_word,
            "question_char": self.question_char,
            "question_len": self.question_len,
            "answer_start": self.answer_start,
            "answer_end": self.answer_end,
            "training": self.training
        }
        if self.use_elmo:
            input_dict['context_tokens'] = self.context_tokens
            input_dict['question_tokens'] = self.question_tokens
        if self.enable_na_answer:
            input_dict["is_impossible"] = self.na
        self.input_placeholder_dict = OrderedDict(input_dict)

        output_dict = {
            "start_prob": self.start_prob,
            "end_prob": self.end_prob
        }
        if self.enable_na_answer:
            output_dict['na_prob'] = self.na_prob*self.na_prob2

        self.output_variable_dict = OrderedDict(output_dict)

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
        na_prob = {}
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
            if not self.enable_na_answer:
                preds_dict[instance['qid']] = pred_answer
            else:
                preds_dict[instance['qid']] = pred_answer if max_prob > output['na_prob'][i] else ''
                na_prob[instance['qid']] = output['na_prob'][i]

        return preds_dict if not self.enable_na_answer else (preds_dict,na_prob)

