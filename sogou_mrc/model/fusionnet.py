# coding:utf-8

from sogou_mrc.model.base_model import BaseModel
from sogou_mrc.nn.similarity_function import ProjectedDotProduct, SymmetricProject
from sogou_mrc.nn.attention import UniAttention, SelfAttn
from sogou_mrc.nn.recurrent import CudnnBiLSTM
from sogou_mrc.nn.layers import VariationalDropout, ElmoEmbedding, PartiallyTrainableEmbedding, Embedding, CoveEmbedding
from sogou_mrc.nn.ops import masked_softmax, mask_logits
from collections import OrderedDict
import tensorflow as tf


class FusionNet(BaseModel):
    def __init__(self, vocab, pretrained_word_embedding=None, word_embedding_size=300, hidden_size=125,
                 dropout_keep_prob=0.6,
                 word_embedding_trainiable=False, finetune_word_size=1000, features=[], feature_vocab={},
                 use_cove_emb=False, cove_path='Keras_CoVe.h5'):
        super(FusionNet, self).__init__(vocab)
        self.use_outer_embedding = False
        self.word_embedding_size = word_embedding_size
        self.vocab = vocab
        self.attention_hidden_size = 250
        self.rnn_hidden_size = hidden_size
        self.pretrained_word_embedding = pretrained_word_embedding
        self.word_embedding_trainiable = word_embedding_trainiable
        self.finetune_word_size = finetune_word_size
        self.dropout_keep_prob = dropout_keep_prob
        self.pos_embedding_size = 12
        self.features = features
        self.feature_vocab = feature_vocab
        self.use_cove_emb = use_cove_emb
        self.ner_embedding_size = 8
        self.cove_path = cove_path
        self._build_graph()

    def _build_graph(self):
        # build input

        self.context_word = tf.placeholder(tf.int32, [None, None])
        self.context_len = tf.placeholder(tf.int32, [None])
        self.question_word = tf.placeholder(tf.int32, [None, None])
        self.question_len = tf.placeholder(tf.int32, [None])
        self.answer_start = tf.placeholder(tf.int32, [None])
        self.answer_end = tf.placeholder(tf.int32, [None])
        self.training = tf.placeholder(tf.bool, [])
        self.match_lemma = tf.placeholder(tf.int32, [None, None])
        self.match_lower = tf.placeholder(tf.int32, [None, None])
        self.pos_feature = tf.placeholder(tf.int32, [None, None])
        self.ner_feature = tf.placeholder(tf.int32, [None, None])
        self.normalized_tf = tf.placeholder(tf.int32, [None, None])

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
            pos_embedding = Embedding(pretrained_embedding=None,
                                      embedding_shape=(len(self.feature_vocab['pos']) + 1, self.pos_embedding_size)
                                      )
            context_pos_feature = pos_embedding(self.pos_feature)
        if len(self.features) > 0 and 'ner' in self.features:
            ner_embedding = Embedding(pretrained_embedding=None,
                                      embedding_shape=(
                                          len(self.feature_vocab['ner']) + 1, self.ner_embedding_size)
                                      )
            context_ner_feature = ner_embedding(self.ner_feature)
        # add dropout
        dropout = VariationalDropout(self.dropout_keep_prob)
        context_word_repr = dropout(context_word_repr, self.training)
        question_word_repr = dropout(question_word_repr, self.training)
        glove_word_repr = context_word_repr
        glove_question_repr = question_word_repr
        if self.use_cove_emb and self.cove_path is not None:
            cove_embedding = CoveEmbedding(cove_path=self.cove_path,
                                           pretrained_word_embedding=self.pretrained_word_embedding, vocab=self.vocab)
            cove_context_repr = cove_embedding(self.context_word, self.context_len)
            cove_question_repr = cove_embedding(self.question_word, self.question_len)
            cove_context_repr = dropout(cove_context_repr, self.training)
            cove_question_repr = dropout(cove_question_repr, self.training)
        # exact_match feature
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
        # features

        if len(self.features) > 0 and 'pos' in self.features:
            context_word_repr = tf.concat([context_word_repr, context_pos_feature], axis=-1)
        if len(self.features) > 0 and 'ner' in self.features:
            context_word_repr = tf.concat([context_word_repr, context_ner_feature], axis=-1)
        if len(self.features) > 0 and 'context_tf' in self.features:
            context_word_repr = tf.concat(
                [context_word_repr, tf.cast(tf.expand_dims(self.normalized_tf, axis=-1), tf.float32)], axis=-1)

        if self.use_cove_emb:
            context_word_repr = tf.concat([cove_context_repr, context_word_repr], axis=-1)
            question_word_repr = tf.concat([cove_question_repr, question_word_repr], axis=-1)

        # 1.2word fusion
        sim_function = ProjectedDotProduct(self.rnn_hidden_size, activation=tf.nn.relu, reuse_weight=True)
        word_fusion = UniAttention(sim_function)
        context2question_fusion = word_fusion(glove_word_repr, glove_question_repr, self.question_len)
        enhanced_context_repr = tf.concat([context_word_repr, exact_match_feature, context2question_fusion], axis=-1)

        enhanced_context_repr = dropout(enhanced_context_repr, self.training)
        # 1.3.1 context encoder
        context_encoder = [CudnnBiLSTM(self.rnn_hidden_size), CudnnBiLSTM(self.rnn_hidden_size)]
        context_low_repr, _ = context_encoder[0](enhanced_context_repr, self.context_len)
        context_low_repr = dropout(context_low_repr, self.training)
        context_high_repr, _ = context_encoder[1](context_low_repr, self.context_len)
        context_high_repr = dropout(context_high_repr, self.training)
        # 1.3.2 question encoder

        question_encoder = [CudnnBiLSTM(self.rnn_hidden_size), CudnnBiLSTM(self.rnn_hidden_size)]
        question_low_repr, _ = question_encoder[0](question_word_repr, self.question_len)
        question_low_repr = dropout(question_low_repr, self.training)
        question_high_repr, _ = question_encoder[1](question_low_repr, self.question_len)
        question_high_repr = dropout(question_high_repr, self.training)
        # 1.4 question understanding
        question_understanding_encoder = CudnnBiLSTM(self.rnn_hidden_size)
        question_understanding, _ = question_understanding_encoder(
            tf.concat([question_low_repr, question_high_repr], axis=-1), self.question_len)
        question_understanding = dropout(question_understanding, self.training)

        # history of context
        context_history = tf.concat([glove_word_repr, context_low_repr, context_high_repr], axis=-1)

        # histor of question
        question_history = tf.concat([glove_question_repr, question_low_repr, question_high_repr], axis=-1)

        # concat cove emb
        if self.use_cove_emb:
            context_history = tf.concat([cove_context_repr, context_history], axis=-1)
            question_history = tf.concat([cove_question_repr, question_history], axis=-1)

        # 1.5.1 low level fusion
        low_level_attn = UniAttention(SymmetricProject(self.attention_hidden_size), name='low_level_fusion')
        low_level_fusion = low_level_attn(context_history, question_history, self.question_len, question_low_repr)
        low_level_fusion = dropout(low_level_fusion, self.training)

        # 1.5.2 high level fusion
        high_level_attn = UniAttention(SymmetricProject(self.attention_hidden_size, name='high_sim_func'),
                                       name='high_level_fusion')
        high_level_fusion = high_level_attn(context_history, question_history, self.question_len, question_high_repr)
        high_level_fusion = dropout(high_level_fusion, self.training)

        # 1.5.3 understanding level fusion
        understanding_attn = UniAttention(
            SymmetricProject(self.attention_hidden_size, name='understanding_sim_func'),
            name='understanding_level_fusion')
        understanding_fusion = understanding_attn(context_history, question_history, self.question_len,
                                                  question_understanding)
        understanding_fusion = dropout(understanding_fusion, self.training)

        # merge context attention
        fully_aware_encoder = CudnnBiLSTM(self.rnn_hidden_size)
        full_fusion_context = tf.concat(
            [context_low_repr, context_high_repr, low_level_fusion, high_level_fusion, understanding_fusion], axis=-1)
        full_fusion_context_repr, _ = fully_aware_encoder(full_fusion_context, self.context_len)

        # history of context
        context_history = tf.concat([glove_word_repr, full_fusion_context, full_fusion_context_repr], axis=-1)
        if self.use_cove_emb:
            context_history = tf.concat([cove_context_repr, context_history], axis=-1)

        # 1.6 self boosted fusion
        self_boosted_attn = UniAttention(SymmetricProject(self.attention_hidden_size, name='self_boosted_attn'),
                                         name='boosted_fusion')
        boosted_fusion = self_boosted_attn(context_history, context_history, self.context_len, full_fusion_context_repr)
        boosted_fusion = dropout(boosted_fusion, self.training)

        # 1.7 context vectors
        context_final_encoder = CudnnBiLSTM(self.rnn_hidden_size)
        context_repr, _ = context_final_encoder(tf.concat([full_fusion_context_repr, boosted_fusion], axis=-1),
                                                self.context_len)
        context_repr = dropout(context_repr, self.training)
        self_attn = SelfAttn()
        U_Q = self_attn(question_understanding, self.question_len)

        # start project
        start_project = tf.keras.layers.Dense(self.rnn_hidden_size * 2, use_bias=False)
        context_repr = dropout(context_repr, self.training)
        start_logits = tf.squeeze(tf.matmul(start_project(context_repr), tf.expand_dims(U_Q, axis=-1)), axis=-1)
        start_logits_masked = mask_logits(start_logits, self.context_len)
        self.start_prob = tf.nn.softmax(start_logits_masked)

        gru_input = tf.reduce_sum(tf.expand_dims(self.start_prob, axis=-1) * context_repr, axis=1)
        GRUCell = tf.contrib.rnn.GRUCell(self.rnn_hidden_size * 2)

        V_Q, _ = GRUCell(gru_input, U_Q)

        # end project
        end_project = tf.keras.layers.Dense(self.rnn_hidden_size * 2, use_bias=False)
        end_logits = tf.squeeze(tf.matmul(end_project(context_repr), tf.expand_dims(V_Q, axis=-1)), axis=-1)
        end_logits_masked = mask_logits(end_logits, self.context_len)
        self.end_prob = tf.nn.softmax(end_logits_masked)
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
            "training": self.training,

        }
        if len(self.features) > 0 and 'match_lower' in self.features:
            input_dict['match_lower'] = self.match_lower
        if len(self.features) > 0 and 'match_lemma' in self.features:
            input_dict['match_lemma'] = self.match_lemma

        if self.use_outer_embedding:
            input_dict['context'] = self.context_string,
            input_dict['question'] = self.question_string
        if len(self.features) > 0 and 'pos' in self.features:
            input_dict['pos'] = self.pos_feature
        if len(self.features) > 0 and 'ner' in self.features:
            input_dict['ner'] = self.ner_feature
        if len(self.features) > 0 and 'context_tf' in self.features:
            input_dict['context_tf'] = self.normalized_tf

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
                    if end_idx <= start_idx + max_len and start_prob * end_prob > max_prob and end_idx >= start_idx:
                        start_position = start_idx
                        end_position = end_idx
                        max_prob = start_prob * end_prob
            char_start_position = instance["context_token_spans"][start_position][0]
            char_end_position = instance["context_token_spans"][end_position][1]
            pred_answer = instance["context"][char_start_position:char_end_position]
            answer_list.append(pred_answer)
        return answer_list
