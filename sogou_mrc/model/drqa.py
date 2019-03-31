# coding:utf-8
from sogou_mrc.model.base_model import BaseModel
from sogou_mrc.nn.similarity_function import ProjectedDotProduct
from sogou_mrc.nn.attention import UniAttention, SelfAttn
from sogou_mrc.nn.recurrent import CudnnBiLSTM
from sogou_mrc.nn.layers import MultiLayerRNN, VariationalDropout, PartiallyTrainableEmbedding, Embedding, ElmoEmbedding
from sogou_mrc.nn.ops import masked_softmax, mask_logits
import numpy as np
from collections import OrderedDict
import tensorflow as tf
from sogou_mrc.train.trainer import Trainer


class DrQA(BaseModel):
    def __init__(self, vocab, pretrained_word_embedding=None, word_embedding_size=300, char_embedding_size=100,
                 char_conv_filters=100,
                 char_conv_kernel_size=5, rnn_hidden_size=128,
                 dropout_keep_prob=0.6, word_embedding_trainiable=False, finetune_word_size=1000, features=[],
                 feature_vocab={}, use_elmo=False, elmo_local_path=None):
        super(DrQA, self).__init__(vocab)
        self.doc_rnn_layers = 3
        self.finetune_word_size = finetune_word_size
        self.question_rnn_layers = 3
        self.rnn_hidden_size = rnn_hidden_size
        self.word_embedding_size = word_embedding_size
        self.keep_prob = dropout_keep_prob
        self.pretrained_word_embedding = pretrained_word_embedding
        self.char_embedding_size = char_embedding_size
        self.char_conv_filters = char_conv_filters
        self.char_conv_kernel_size = char_conv_kernel_size
        self.word_embedding_trainiable = word_embedding_trainiable
        self.features = features
        self.feature_vocab = feature_vocab
        self.use_elmo = use_elmo
        self.elmo_local_path = elmo_local_path
        self._build_graph()

    def _build_graph(self):
        # build input

        self.context_word = tf.placeholder(tf.int32, [None, None])
        self.context_len = tf.placeholder(tf.int32, [None])
        self.question_word = tf.placeholder(tf.int32, [None, None])
        self.match_lemma = tf.placeholder(tf.int32, [None, None])
        self.match_lower = tf.placeholder(tf.int32, [None, None])
        self.question_len = tf.placeholder(tf.int32, [None])
        self.answer_start = tf.placeholder(tf.int32, [None])
        self.answer_end = tf.placeholder(tf.int32, [None])
        self.pos_feature = tf.placeholder(tf.int32, [None, None])
        self.ner_feature = tf.placeholder(tf.int32, [None, None])
        self.normalized_tf = tf.placeholder(tf.int32, [None, None])
        self.question_tokens = tf.placeholder(tf.string, [None, None])
        self.context_tokens = tf.placeholder(tf.string, [None, None])

        self.training = tf.placeholder(tf.bool, [])

        # 1. Word encoding
        word_embedding = PartiallyTrainableEmbedding(
            trainable_num=self.finetune_word_size,
            pretrained_embedding=self.pretrained_word_embedding,
            embedding_shape=(len(self.vocab.get_word_vocab()) + 1, self.word_embedding_size))

        # 1.1 Embedding
        context_word_repr = word_embedding(self.context_word)
        question_word_repr = word_embedding(self.question_word)

        # pos embedding
        if len(self.features) > 0 and 'pos' in self.features:
            self.context_pos_feature = tf.cast(tf.one_hot(self.pos_feature, len(self.feature_vocab['pos']) + 1),
                                               tf.float32)
        # ner embedding
        if len(self.features) > 0 and 'ner' in self.features:
            self.context_ner_feature = tf.cast(tf.one_hot(self.ner_feature, len(self.feature_vocab['ner']) + 1),
                                               tf.float32)
        dropout = VariationalDropout(self.keep_prob)
        # embedding dropout
        context_word_repr = dropout(context_word_repr, self.training)
        question_word_repr = dropout(question_word_repr, self.training)

        # elmo embedding
        if self.use_elmo:
            elmo_emb = ElmoEmbedding(local_path=self.elmo_local_path)
            context_elmo_repr = elmo_emb(self.context_tokens, self.context_len)
            context_elmo_repr = dropout(context_elmo_repr, self.training)
            question_elmo_repr = elmo_emb(self.question_tokens, self.question_len)
            question_elmo_repr = dropout(question_elmo_repr, self.training)
            context_word_repr = tf.concat([context_word_repr, context_elmo_repr], axis=-1)
            question_word_repr = tf.concat([question_word_repr, question_elmo_repr], axis=-1)

        # 1.2 exact match feature
        context_expanded = tf.tile(tf.expand_dims(self.context_word, axis=-1), [1, 1, tf.shape(self.question_word)[1]])
        query_expanded = tf.tile(tf.expand_dims(self.question_word, axis=1), [1, tf.shape(self.context_word)[1], 1])
        exact_match_feature = tf.cast(tf.reduce_any(tf.equal(context_expanded, query_expanded), axis=-1), tf.float32)
        exact_match_feature = tf.expand_dims(exact_match_feature, axis=-1)
        if len(self.features) > 0 and 'match_lower' in self.features:
            match_lower_feature = tf.expand_dims(tf.cast(self.match_lower, tf.float32), axis=-1)
            exact_match_feature = tf.concat([exact_match_feature, match_lower_feature], axis=-1)
        if len(self.features) > 0 and 'match_lemma' in self.features:
            exact_match_feature = tf.concat(
                [exact_match_feature, tf.expand_dims(tf.cast(self.match_lemma, tf.float32), axis=-1)], axis=-1)
        if len(self.features) > 0 and 'pos' in self.features:
            exact_match_feature = tf.concat([exact_match_feature, self.context_pos_feature], axis=-1)
        if len(self.features) > 0 and 'ner' in self.features:
            exact_match_feature = tf.concat([exact_match_feature, self.context_ner_feature], axis=-1)
        if len(self.features) > 0 and 'context_tf' in self.features:
            exact_match_feature = tf.concat(
                [exact_match_feature, tf.cast(tf.expand_dims(self.normalized_tf, axis=-1), tf.float32)], axis=-1)
        # 1.3 aligned question embedding
        sim_function = ProjectedDotProduct(self.rnn_hidden_size, activation=tf.nn.relu, reuse_weight=True)
        word_fusion = UniAttention(sim_function)
        aligned_question_embedding = word_fusion(context_word_repr, question_word_repr, self.question_len)

        # context_repr
        context_repr = tf.concat([context_word_repr, exact_match_feature, aligned_question_embedding], axis=-1)

        # 1.4context encoder
        context_rnn_layers = [CudnnBiLSTM(self.rnn_hidden_size) for _ in range(self.doc_rnn_layers)]
        multi_bilstm_layer_context = MultiLayerRNN(context_rnn_layers, concat_layer_out=True, input_keep_prob=0.7)
        context_repr = multi_bilstm_layer_context(context_repr, self.context_len, self.training)
        # rnn output dropout
        context_repr = dropout(context_repr, self.training)

        # 1.5 question encoder
        question_rnn_layers = [CudnnBiLSTM(self.rnn_hidden_size) for _ in range(self.question_rnn_layers)]
        multi_bilstm_layer_question = MultiLayerRNN(question_rnn_layers, concat_layer_out=True, input_keep_prob=0.7)
        question_repr = multi_bilstm_layer_question(question_word_repr, self.question_len, self.training)
        # rnn output dropout
        question_repr = dropout(question_repr, self.training)
        self_attn = SelfAttn()
        weighted_question_repr = self_attn(question_repr, self.question_len)

        # predict
        doc_hidden_size = self.rnn_hidden_size * self.doc_rnn_layers * 2
        start_project = tf.keras.layers.Dense(doc_hidden_size, use_bias=False)
        start_logits = tf.squeeze(
            tf.matmul(start_project(context_repr), tf.expand_dims(weighted_question_repr, axis=-1)), axis=-1)
        self.start_prob = masked_softmax(start_logits, self.context_len)
        end_project = tf.keras.layers.Dense(doc_hidden_size, use_bias=False)
        end_logits = tf.squeeze(tf.matmul(end_project(context_repr), tf.expand_dims(weighted_question_repr, axis=-1)),
                                axis=-1)
        self.end_prob = masked_softmax(end_logits, self.context_len)
        # 7. Loss and input/output dict
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
            "context_len": self.context_len,
            "question_word": self.question_word,
            "question_len": self.question_len,
            "answer_start": self.answer_start,
            "answer_end": self.answer_end,
            "training": self.training
        }
        if len(self.features) > 0 and 'match_lemma' in self.features:
            input_dict['match_lemma'] = self.match_lemma
        if len(self.features) > 0 and 'context_tf' in self.features:
            input_dict['context_tf'] = self.normalized_tf
        if len(self.features) > 0 and 'match_lower' in self.features:
            input_dict['match_lower'] = self.match_lower
        if len(self.features) > 0 and 'pos' in self.features:
            input_dict['pos'] = self.pos_feature
        if len(self.features) > 0 and 'ner' in self.features:
            input_dict['ner'] = self.ner_feature
        if self.use_elmo:
            input_dict['context_tokens'] = self.context_tokens
            input_dict['question_tokens'] = self.question_tokens
        self.input_placeholder_dict = OrderedDict(input_dict)

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

    def compile(self):
        # adamax for fast coverage
        params = tf.trainable_variables()
        param_with_gradient = []
        for grad, param in zip(tf.gradients(self.loss, params), params):
            if grad is not None: param_with_gradient.append(param)
        self.train_op = tf.contrib.keras.optimizers.Adamax().get_updates(loss=self.loss, params=param_with_gradient)

    def get_best_answer(self, output, instances, max_len=15):
        answer_list = []
        for i in range(len(output['start_prob'])):
            instance = instances[i]
            max_prob = 0.0
            start_position = 0
            end_position = 0
            for start_idx, start_prob in enumerate(output['start_prob'][i]):
                for end_idx, end_prob in enumerate(output['end_prob'][i]):
                    if end_idx <= start_idx + max_len and start_prob * end_prob > max_prob and start_idx <= end_idx:
                        start_position = start_idx
                        end_position = end_idx
                        max_prob = start_prob * end_prob
            char_start_position = instance["context_token_spans"][start_position][0]
            char_end_position = instance["context_token_spans"][end_position][1]
            pred_answer = instance["context"][char_start_position:char_end_position]
            answer_list.append(pred_answer)
        return answer_list
