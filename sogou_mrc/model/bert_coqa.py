# coding:utf-8
import os
import tensorflow as tf
import numpy as np
import logging
from collections import OrderedDict, defaultdict
from sogou_mrc.model.base_model import BaseModel
from sogou_mrc.train.trainer import Trainer
# from sogou_mrc.libraries.BertWrapper import BertModelHelper
from sogou_mrc.nn.layers import BertEmbedding
from sogou_mrc.libraries import modeling
from sogou_mrc.libraries import optimization

VERY_NEGATIVE_NUMBER = -1e29


class BertCoQA(BaseModel):
    def __init__(self, vocab=None, bert_dir='', answer_verification=True):
        super(BertCoQA, self).__init__(vocab)
        self.bert_dir = bert_dir
        self.activation = 'relu'
        self.answer_verification = answer_verification
        self.beta = 100
        self.n_layers = 2
        self._build_graph()

    def _build_graph(self):
        self.training = tf.placeholder(tf.bool, shape=())
        self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.segment_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.question_len = tf.placeholder(shape=[None], dtype=tf.int32)
        self.start_position = tf.placeholder(shape=[None], dtype=tf.int32)
        self.end_position = tf.placeholder(shape=[None], dtype=tf.int32)
        self.question_mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.yes_masks = tf.placeholder(shape=[None], dtype=tf.int32)
        self.unk_masks = tf.placeholder(shape=[None], dtype=tf.int32)
        self.no_masks = tf.placeholder(shape=[None], dtype=tf.int32)
        self.rationale_mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.extractive_masks = tf.placeholder(shape=[None], dtype=tf.int32)
        self.bert_embedding = BertEmbedding(self.bert_dir)
        final_hidden, pooled_output = self.bert_embedding(input_ids=self.input_ids, input_mask=self.input_mask,
                                                          segment_ids=self.segment_ids, is_training=self.training,
                                                          use_one_hot_embeddings=False,
                                                          return_pool_output=True)
        final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
        batch_size = final_hidden_shape[0]
        seq_length = final_hidden_shape[1]
        hidden_size = final_hidden_shape[2]
        final_hidden_matrix = tf.reshape(final_hidden,
                                         [batch_size * seq_length, hidden_size])

        rationale_logits = None

        if self.answer_verification:
            with tf.variable_scope("rationale"):
                rationale_logits = self.multi_linear_layer(final_hidden_matrix, self.n_layers, hidden_size, 1,
                                                           activation=self.activation)  # batch*seq_len, 1
            rationale_logits = tf.nn.sigmoid(rationale_logits)  # batch*seq_len, 1
            rationale_logits = tf.reshape(rationale_logits, [batch_size, seq_length])  # batch, seq_len
            segment_mask = tf.cast(self.segment_ids, tf.float32)  # batch, seq_len      
          
            final_hidden = final_hidden * tf.expand_dims(rationale_logits, 2)
            final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
            with tf.variable_scope("answer_logits"):
                logits = self.multi_linear_layer(final_hidden_matrix, self.n_layers, hidden_size, 2,
                                                 activation=self.activation)
            logits = tf.reshape(logits, [batch_size, seq_length, 2])
            logits = tf.transpose(logits, [2, 0, 1])

            unstacked_logits = tf.unstack(logits, axis=0)
            (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

            rationale_logits = rationale_logits * segment_mask
            start_logits = start_logits * rationale_logits
            end_logits = end_logits * rationale_logits

            # print(unk_yes_no_logits)
            with tf.variable_scope("unk"):
                unk_logits = self.multi_linear_layer(pooled_output, self.n_layers, hidden_size, 1,
                                                     activation=self.activation)

            with tf.variable_scope("doc_attn"):
                attention = self.multi_linear_layer(final_hidden_matrix, self.n_layers, hidden_size, 1,
                                                    activation=self.activation)  # batch*seq_len, 1
            attention = tf.reshape(attention, [batch_size, seq_length])  # batch, seq_len
            attention = attention * tf.cast(self.input_mask, tf.float32) + tf.cast((1 - self.input_mask),
                                                                                   tf.float32) * VERY_NEGATIVE_NUMBER
            attention = tf.nn.softmax(attention, 1)  # batch, seq_len
            attention = tf.expand_dims(attention, 2)  # batch, seq_len, 1
            attention_pooled_output = tf.reduce_sum(attention * final_hidden, 1)  # batch, hidden_size
            with tf.variable_scope("yes_no"):
                yes_no_logits = self.multi_linear_layer(attention_pooled_output, self.n_layers, hidden_size, 2,
                                                        activation=self.activation)
            unstacked_logits1 = tf.unstack(yes_no_logits, axis=1)
            yes_logits, no_logits = unstacked_logits1

            yes_logits = tf.expand_dims(yes_logits, 1)
            no_logits = tf.expand_dims(no_logits, 1)         
        else:
            with tf.variable_scope("answer_logits"):
                logits = self.multi_linear_layer(final_hidden_matrix, self.n_layers, hidden_size, 2,
                                                 activation=self.activation)
            logits = tf.reshape(logits, [batch_size, seq_length, 2])
            logits = tf.transpose(logits, [2, 0, 1])

            unstacked_logits = tf.unstack(logits, axis=0)
            (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
        
            # print(unk_yes_no_logits)
            with tf.variable_scope("unk_yes_no"):
                unk_yes_no_logits = self.multi_linear_layer(pooled_output, self.n_layers, hidden_size, 3,
                                                            activation="relu")
            unstacked_logits1 = tf.unstack(unk_yes_no_logits, axis=1)
            unk_logits, yes_logits, no_logits = unstacked_logits1

            unk_logits = tf.expand_dims(unk_logits, 1)
            yes_logits = tf.expand_dims(yes_logits, 1)
            no_logits = tf.expand_dims(no_logits, 1)

        input_mask0 = tf.cast(self.input_mask, tf.float32)
        masked_start_logits = start_logits * input_mask0 + (1 - input_mask0) * VERY_NEGATIVE_NUMBER
        masked_end_logits = end_logits * input_mask0 + (1 - input_mask0) * VERY_NEGATIVE_NUMBER

        seq_length = modeling.get_shape_list(self.input_ids)[1]  # input_mask0 = tf.cast(self.input_mask, tf.float32)
        start_masks = tf.one_hot(self.start_position, depth=seq_length, dtype=tf.float32)
        end_masks = tf.one_hot(self.end_position, depth=seq_length, dtype=tf.float32)
        start_masks = start_masks * tf.expand_dims(tf.cast(self.extractive_masks, tf.float32), axis=-1)
        end_masks = end_masks * tf.expand_dims(tf.cast(self.extractive_masks, tf.float32), axis=-1)

        unk_masks = tf.expand_dims(tf.cast(self.unk_masks, tf.float32), axis=-1)
        yes_masks = tf.expand_dims(tf.cast(self.yes_masks, tf.float32), axis=-1)
        no_masks = tf.expand_dims(tf.cast(self.no_masks, tf.float32), axis=-1)
        new_start_masks = tf.concat([start_masks, unk_masks, yes_masks, no_masks], axis=1)
        new_end_masks = tf.concat([end_masks, unk_masks, yes_masks, no_masks], axis=1)

        new_start_logits = tf.concat([masked_start_logits, unk_logits, yes_logits, no_logits], axis=1)
        new_end_logits = tf.concat([masked_end_logits, unk_logits, yes_logits, no_logits], axis=1)

        start_log_norm = tf.reduce_logsumexp(new_start_logits, axis=1)
        start_log_score = tf.reduce_logsumexp(
            new_start_logits + VERY_NEGATIVE_NUMBER * (1 - tf.cast(new_start_masks, tf.float32)), axis=1)
        start_loss = tf.reduce_mean(-(start_log_score - start_log_norm))

        end_log_norm = tf.reduce_logsumexp(new_end_logits, axis=1)
        end_log_score = tf.reduce_logsumexp(
            new_end_logits + VERY_NEGATIVE_NUMBER * (1 - tf.cast(new_end_masks, tf.float32)), axis=1)
        end_loss = tf.reduce_mean(-(end_log_score - end_log_norm))

        if self.answer_verification:
            rationale_positions = self.rationale_mask
            alpha = 0.25
            gamma = 2.
            rationale_positions = tf.cast(rationale_positions, tf.float32)
            rationale_loss = -alpha * ((1 - rationale_logits) ** gamma) * rationale_positions * tf.log(
                rationale_logits + 1e-8) - (1 - alpha) * (rationale_logits ** gamma) * (
                                     1 - rationale_positions) * tf.log(1 - rationale_logits + 1e-8)
            rationale_loss = tf.reduce_sum(rationale_loss * segment_mask) / tf.reduce_sum(segment_mask)
            total_loss = (start_loss + end_loss) / 2.0 + rationale_loss * self.beta
        else:
            total_loss = (start_loss + end_loss) / 2.0

        self.loss = total_loss

        self.input_placeholder_dict = OrderedDict({
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "segment_ids": self.segment_ids,
            "training": self.training,
            "start_position": self.start_position,
            'end_position': self.end_position,
            "unk_mask": self.unk_masks,
            "yes_mask": self.yes_masks,
            "no_mask": self.no_masks,
            "rationale_mask": self.rationale_mask,
            "extractive_mask": self.extractive_masks,
        })

        self.output_variable_dict = OrderedDict({
            "start_logits": start_logits,
            "end_logits": end_logits,
            "unk_logits": unk_logits,
            "yes_logits": yes_logits,
            "no_logits": no_logits
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

    def multi_linear_layer(self, x, layers, hidden_size, output_size, activation=None):

        if layers <= 0:
            return x
        for i in range(layers - 1):
            with tf.variable_scope("linear_layer" + str(i + 1)):
                w = tf.get_variable(
                    "w", [hidden_size, hidden_size],
                    initializer=tf.truncated_normal_initializer(stddev=0.02))

                b = tf.get_variable(
                    "b", [hidden_size], initializer=tf.zeros_initializer())
                x = tf.nn.bias_add(tf.matmul(x, w), b)
                if activation == "relu":
                    x = tf.nn.relu(x)
                elif activation == "tanh":
                    x = tf.tanh(x)
                elif activation == "gelu":
                    x = modeling.gelu(x)
        with tf.variable_scope("linear_layer" + str(layers)):
            w = tf.get_variable(
                "w", [hidden_size, output_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            b = tf.get_variable(
                "b", [output_size], initializer=tf.zeros_initializer())
            x = tf.nn.bias_add(tf.matmul(x, w), b)
        return x

    def compile(self, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False):
        self.train_op = optimization.create_optimizer(self.loss, learning_rate, num_train_steps, num_warmup_steps,
                                                      use_tpu)

    def train_and_evaluate(self, train_generator, eval_generator, evaluator, epochs=1, eposides=1,
                           save_dir=None, summary_dir=None, save_summary_steps=10):
        if not self.initialized:
            self.bert_embedding.init_bert()
            self.session.run(tf.global_variables_initializer())

        Trainer._train_and_evaluate(self, train_generator, eval_generator, evaluator, epochs=epochs,
                                    eposides=eposides,
                                    save_dir=save_dir, summary_dir=summary_dir, save_summary_steps=save_summary_steps)

    def get_best_answer(self, output, instances, max_answer_len=11, null_score_diff_threshold=0.0):
        def _get_best_indexes(logits, n_best_size):
            """Get the n-best logits from a list."""
            index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

            best_indexes = []
            for i in range(len(index_and_score)):
                if i >= n_best_size:
                    break
                best_indexes.append(index_and_score[i][0])
            return best_indexes

        ground_answers = []
        qid_with_max_logits = {}
        qid_with_final_text = {}
        qid_with_no_logits = {}
        qid_with_yes_logits = {}
        qid_with_unk_logits = {}
        for i in range(len(instances)):
            instance = instances[i]
            ground_answers.append(instance['answer'])
            start_logits = output['start_logits'][i]
            end_logits = output['end_logits'][i]
            feature_unk_score = output['unk_logits'][i][0] * 2
            feature_yes_score = output['yes_logits'][i][0] * 2
            feature_no_score = output['no_logits'][i][0] * 2
            start_indexes = _get_best_indexes(start_logits, n_best_size=20)
            end_indexes = _get_best_indexes(end_logits, n_best_size=20)
            max_start_index = -1
            max_end_index = -1
            max_logits = -100000000
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(instance['tokens']):
                        continue
                    if end_index >= len(instance['tokens']):
                        continue
                    if start_index not in instance['token_to_orig_map']:
                        continue
                    if end_index not in instance['token_to_orig_map']:
                        continue
                    if end_index < start_index:
                        continue
                    if not instance['token_is_max_context'].get(start_index, False):
                        continue
                    length = end_index - start_index - 1
                    if length > max_answer_len:
                        continue
                    sum_logits = start_logits[start_index] + end_logits[end_index]
                    if sum_logits > max_logits:
                        max_logits = sum_logits
                        max_start_index = start_index
                        max_end_index = end_index
            final_text = ''
            if (max_start_index != -1 and max_end_index != -1):
                final_text = self.prediction_to_ori(max_start_index, max_end_index, instance)
            story_id, turn_id = instance["qid"].split("|")
            turn_id = int(turn_id)
            if (story_id, turn_id) in qid_with_max_logits and max_logits > qid_with_max_logits[(story_id, turn_id)]:
                qid_with_max_logits[(story_id, turn_id)] = max_logits
                qid_with_final_text[(story_id, turn_id)] = final_text
            if (story_id, turn_id) not in qid_with_max_logits:
                qid_with_max_logits[(story_id, turn_id)] = max_logits
                qid_with_final_text[(story_id, turn_id)] = final_text
            if (story_id, turn_id) not in qid_with_no_logits:
                qid_with_no_logits[(story_id, turn_id)] = feature_no_score
            if feature_no_score > qid_with_no_logits[(story_id, turn_id)]:
                qid_with_no_logits[(story_id, turn_id)] = feature_no_score
            if (story_id, turn_id) not in qid_with_yes_logits:
                qid_with_yes_logits[(story_id, turn_id)] = feature_yes_score
            if feature_yes_score > qid_with_yes_logits[(story_id, turn_id)]:
                qid_with_yes_logits[(story_id, turn_id)] = feature_yes_score
            if (story_id, turn_id) not in qid_with_unk_logits:
                qid_with_unk_logits[(story_id, turn_id)] = feature_unk_score
            if feature_unk_score > qid_with_unk_logits[(story_id, turn_id)]:
                qid_with_unk_logits[(story_id, turn_id)] = feature_unk_score
        result = {}
        for k in qid_with_max_logits:
            scores = [qid_with_max_logits[k], qid_with_no_logits[k], qid_with_yes_logits[k], qid_with_unk_logits[k]]
            max_val = max(scores)
            if max_val == qid_with_max_logits[k]:
                result[k] = qid_with_final_text[k]
            elif max_val == qid_with_unk_logits[k]:
                result[k] = 'unknown'
            elif max_val == qid_with_yes_logits[k]:
                result[k] = 'yes'
            else:
                result[k] = 'no'
        return result

    def prediction_to_ori(self, start_index, end_index, instance):
        if start_index > 0:
            tok_tokens = instance['tokens'][start_index:end_index + 1]
            orig_doc_start = instance['token_to_orig_map'][start_index]
            orig_doc_end = instance['token_to_orig_map'][end_index]
            char_start_position = instance["context_token_spans"][orig_doc_start][0]
            char_end_position = instance["context_token_spans"][orig_doc_end][1]
            pred_answer = instance["context"][char_start_position:char_end_position]
            return pred_answer
        return ""

    def evaluate(self, batch_generator, evaluator):
        self.bert_embedding.init_bert()
        Trainer._evaluate(self, batch_generator, evaluator)
