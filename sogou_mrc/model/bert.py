# coding:utf-8
import os
import tensorflow as tf
import numpy as np
import logging
from collections import OrderedDict, defaultdict
from sogou_mrc.model.base_model import BaseModel
from sogou_mrc.train.trainer import Trainer
# from sogou_mrc.libraries.BertWrapper import BertModelHelper
from sogou_mrc.nn.layers import  BertEmbedding
from sogou_mrc.libraries import modeling
from sogou_mrc.libraries import optimization


class BertBaseline(BaseModel):
    def __init__(self, vocab=None, bert_dir='', version_2_with_negative=True):
        super(BertBaseline, self).__init__(vocab)
        self.bert_dir = bert_dir
        self.version_2_with_negative = version_2_with_negative
        self._build_graph()

    def _build_graph(self):
        self.training = tf.placeholder(tf.bool, shape=())
        self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.segment_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.start_position = tf.placeholder(shape=[None], dtype=tf.int32)
        self.end_position = tf.placeholder(shape=[None], dtype=tf.int32)
        self.bert_embedding = BertEmbedding(self.bert_dir)
        final_hidden = self.bert_embedding(input_ids=self.input_ids,input_mask = self.input_mask,segment_ids=self.segment_ids,is_training=self.training,return_pool_output=False)
        final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
        batch_size = final_hidden_shape[0]
        seq_length = final_hidden_shape[1]
        hidden_size = final_hidden_shape[2]

        output_weights = tf.get_variable(
            "cls/squad/output_weights", [2, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

        final_hidden_matrix = tf.reshape(final_hidden,
                                         [batch_size * seq_length, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        logits = tf.reshape(logits, [batch_size, seq_length, 2])
        logits = tf.transpose(logits, [2, 0, 1])

        unstacked_logits = tf.unstack(logits, axis=0)

        start_logits = unstacked_logits[0]
        end_logits = unstacked_logits[1]

        def compute_loss(logits, positions):
            one_hot_positions = tf.one_hot(
                positions, depth=seq_length, dtype=tf.float32)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            loss = -tf.reduce_mean(
                tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
            return loss

        global_step = tf.train.get_or_create_global_step()

        start_loss = compute_loss(start_logits, self.start_position)
        end_loss = compute_loss(end_logits, self.end_position)
        self.loss = (start_loss + end_loss) / 2.0

        self.input_placeholder_dict = OrderedDict({
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "segment_ids": self.segment_ids,
            "training": self.training,
            "start_position": self.start_position,
            'end_position': self.end_position
        })

        self.output_variable_dict = OrderedDict({
            "start_logits": start_logits,
            "end_logits": end_logits
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

    def get_best_answer(self, output, instances, max_answer_len=30, null_score_diff_threshold=0.0):
        def _get_best_indexes(logits, n_best_size):
            """Get the n-best logits from a list."""
            index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

            best_indexes = []
            for i in range(len(index_and_score)):
                if i >= n_best_size:
                    break
                best_indexes.append(index_and_score[i][0])
            return best_indexes

        answer_list = []
        pids_list = []
        ground_answers = []
        pred_dict = {}
        qid_with_max_logits = {}
        qid_with_final_text = {}
        qid_with_null_logits = {}
        na_prob = {}
        for i in range(len(instances)):
            instance = instances[i]
            ground_answers.append(instance['answer'])
            start_logits = output['start_logits'][i]
            end_logits = output['end_logits'][i]
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min mull score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            if self.version_2_with_negative:
                feature_null_score = start_logits[0] + end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    null_start_logit = start_logits[0]
                    null_end_logit = end_logits[0]
            start_indexes = _get_best_indexes(start_logits, n_best_size=20)
            end_indexes = _get_best_indexes(end_logits, n_best_size=20)
            max_start_index = -1
            max_end_index = -1
            import collections
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
            # max_start_index = instance['start_position']
            # max_end_index = instance['end_position']
            import math
            def _compute_softmax(scores):
                """Compute softmax probability over raw logits."""
                if not scores:
                    return []

                max_score = None
                for score in scores:
                    if max_score is None or score > max_score:
                        max_score = score

                exp_scores = []
                total_sum = 0.0
                for score in scores:
                    x = math.exp(score - max_score)
                    exp_scores.append(x)
                    total_sum += x

                probs = []
                for score in exp_scores:
                    probs.append(score / total_sum)
                return probs

            final_text = ''
            if (self.version_2_with_negative and max_start_index != -1 and max_end_index != -1) \
                    or \
                    (max_start_index != -1 and max_end_index != -1):
                final_text = self.prediction_to_ori(max_start_index, max_end_index, instance)
            qid = instance['qid']
            if qid in qid_with_max_logits and max_logits > qid_with_max_logits[qid]:
                qid_with_max_logits[qid] = max_logits
                qid_with_final_text[qid] = final_text
            if qid not in qid_with_max_logits:
                qid_with_max_logits[qid] = max_logits
                qid_with_final_text[qid] = final_text
            if self.version_2_with_negative:
                if qid not in qid_with_null_logits:
                    qid_with_null_logits[qid] = score_null
                if qid in qid_with_null_logits and score_null > qid_with_null_logits[qid]:
                    qid_with_null_logits[qid] = score_null
                if qid_with_null_logits[qid] - qid_with_max_logits[qid] > null_score_diff_threshold:
                    qid_with_final_text[qid] = ""
                na_prob[qid] = qid_with_null_logits[qid] - qid_with_max_logits[qid]
        if not self.version_2_with_negative:
            return qid_with_final_text
        return qid_with_final_text, na_prob

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
