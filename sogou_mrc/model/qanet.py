# coding:utf-8

from sogou_mrc.model.base_model import BaseModel
from sogou_mrc.nn.similarity_function import ProjectedDotProduct, SymmetricProject
from sogou_mrc.nn.attention import SelfAttn
from sogou_mrc.nn.recurrent import CudnnBiLSTM,CudnnBiGRU,CudnnGRU
from sogou_mrc.nn.layers import VariationalDropout, Dropout, Embedding, EncoderBlock, Conv1DAndMaxPooling, Highway
from sogou_mrc.nn.ops import masked_softmax, mask_logits
from sogou_mrc.train.trainer import Trainer
import numpy as np
from collections import OrderedDict,deque
import tensorflow as tf


class QANET(BaseModel):
    def __init__(self, vocab, pretrained_word_embedding=None,word_embedding_size=300,char_embedding_size=8,filters=128,
                 dropout_keep_prob=0.75, word_embedding_trainable=False, ema_decay=0.9999):
        super(QANET, self).__init__(vocab)
        self.kernel_size1, self.kernel_size2 = 5, 7
        self.conv_layers1, self.conv_layers2 = 4, 2
        self.model_encoder_layers = 3
        self.heads = 8
        self.char_filters = 8
        self.filters = filters
        self.word_embedding_size = word_embedding_size
        self.keep_prob = dropout_keep_prob
        self.pretrained_word_embedding = pretrained_word_embedding
        self.char_embedding_size = char_embedding_size
        self.word_embedding_trainable = word_embedding_trainable
        self.ema_decay = ema_decay

        self._build_graph()
        self.initialized = False

    def _build_graph(self):
        # build input
        self.training = tf.placeholder(tf.bool, [])
        self.context_word = tf.placeholder(tf.int32, [None, None])
        self.slice_context_len = tf.where(self.training, 400, 1000)
        self.slice_question_len = tf.where(self.training, 50, 100)
        self.context_char = tf.placeholder(tf.int32, [None, None, None])
        self.context_char = tf.placeholder(tf.int32, [None, None, None])
        batch_size, original_context_len, max_context_word_len = tf.shape(self.context_char)[0],tf.shape(self.context_char)[1],tf.shape(self.context_char)[2]

        slice_context_word = tf.slice(self.context_word,[0,0],[batch_size,tf.minimum(self.slice_context_len,original_context_len)])
        slice_context_char = tf.slice(self.context_char, [0, 0, 0], [batch_size,tf.minimum(self.slice_context_len,original_context_len),max_context_word_len])
        self.context_len = tf.placeholder(tf.int32, [None])
        self.question_word = tf.placeholder(tf.int32, [None, None])
        self.question_char = tf.placeholder(tf.int32, [None, None, None])
        original_question_len,max_question_word_len = tf.shape(self.question_char)[1],tf.shape(self.question_char)[2]
        slice_question_word = tf.slice(self.question_word,[0,0],[batch_size,tf.minimum(self.slice_question_len,original_question_len)])
        slice_question_char = tf.slice(self.question_char, [0, 0, 0],[batch_size, tf.minimum(self.slice_question_len, original_question_len),max_question_word_len])
        self.question_len = tf.placeholder(tf.int32, [None])
        self.answer_start = tf.placeholder(tf.int32, [None])
        self.answer_end = tf.placeholder(tf.int32, [None])

        max_context_len = tf.shape(slice_context_word)[1]
        max_question_len = tf.shape(slice_question_word)[1]
        slice_context_len = tf.clip_by_value(self.context_len,0,max_context_len)
        slice_question_len = tf.clip_by_value(self.question_len,0,max_question_len)
        context_mask = (tf.sequence_mask(slice_context_len, max_context_len, dtype=tf.float32)-1)*100
        question_mask = (tf.sequence_mask(slice_question_len, max_question_len, dtype=tf.float32)-1) * 100
        divisors = tf.pow(tf.constant([10000.0] * (self.filters // 2), dtype=tf.float32),tf.range(0, self.filters, 2, dtype=tf.float32) / self.filters)
        quotients = tf.cast(tf.expand_dims(tf.range(0, max_context_len), -1),tf.float32) / tf.expand_dims(divisors, 0)
        position_repr = tf.concat([tf.sin(quotients), tf.cos(quotients)], -1) #  CL*F
        # 1. Word encoding
        word_embedding = Embedding(pretrained_embedding=self.pretrained_word_embedding,
                                   embedding_shape=(len(self.vocab.get_word_vocab()) + 1, self.word_embedding_size),
                                   trainable=self.word_embedding_trainable)
        char_embedding = Embedding(embedding_shape=(len(self.vocab.get_char_vocab()) + 1, self.char_embedding_size),
                                   trainable=True, init_scale=0.2)
        dropout = Dropout(self.keep_prob)
        # 1.1 Embedding
        context_word_embedding = Dropout(0.9)(word_embedding(slice_context_word),self.training) # B*CL*WD
        context_char_embedding = Dropout(0.95)(char_embedding(slice_context_char),self.training) # B*CL*WL*CD
        question_word_embedding = Dropout(0.9)(word_embedding(slice_question_word),self.training) # B*QL*WD
        question_char_embedding = Dropout(0.95)(char_embedding(slice_question_char),self.training) # B*QL*WL*CD

        char_cnn = Conv1DAndMaxPooling(self.char_filters,self.kernel_size1,padding='same',activation=tf.nn.relu)
        embedding_dense = tf.keras.layers.Dense(self.filters)
        highway = Highway(affine_activation=tf.nn.relu, trans_gate_activation=tf.nn.sigmoid,
                          hidden_units=self.filters, keep_prob=self.keep_prob)

        context_char_repr = tf.reshape(char_cnn(context_char_embedding),[-1,max_context_len,self.char_filters]) # B*CL*CF
        context_repr = highway(dropout(embedding_dense(tf.concat([context_word_embedding,context_char_repr],-1)), self.training),self.training) # B*CL*F

        question_char_repr = tf.reshape(char_cnn(question_char_embedding),[-1,max_question_len,self.char_filters]) # B*QL*CF
        question_repr = highway(dropout(embedding_dense(tf.concat([question_word_embedding,question_char_repr],-1)), self.training),self.training) # B*QL*CF

        # 1.2 Embedding Encoder
        embedding_encoder = EncoderBlock(self.kernel_size1,self.filters,self.conv_layers1,self.heads,self.keep_prob)
        embedding_context = tf.contrib.layers.layer_norm(embedding_encoder(context_repr + position_repr,self.training,context_mask), begin_norm_axis=-1)  # B*CL*F
        embedding_question = tf.contrib.layers.layer_norm(embedding_encoder(question_repr + position_repr[:max_question_len], self.training,question_mask), begin_norm_axis=-1)  # B*QL*F

        # 1.3 co-attention
        co_attention_context = tf.keras.layers.Dense(1)(embedding_context)  # B*CL*1
        co_attention_question = tf.keras.layers.Dense(1)(embedding_question)  # B*QL*1
        cq = tf.matmul(tf.expand_dims(tf.transpose(embedding_context, [0, 2, 1]), -1),tf.expand_dims(tf.transpose(embedding_question, [0, 2, 1]), -2))  # B*F*CL*QL
        cq_score = tf.keras.layers.Dense(1)(tf.transpose(cq, [0, 2, 3, 1]))[:, :, :, 0] + co_attention_context + tf.transpose(co_attention_question, [0, 2, 1])  # B*CL*QL
        question_similarity = tf.nn.softmax(cq_score + tf.expand_dims(question_mask, -2),2)  # B*CL*QL
        context_similarity = tf.nn.softmax(cq_score + tf.expand_dims(context_mask, -1), 1)  # B*CL*QL
        cqa = tf.matmul(question_similarity, embedding_question)  #  B*CL*F
        qca = tf.matmul(question_similarity, tf.matmul(context_similarity, embedding_context, transpose_a=True))  #  B*CL*F
        co_attention_output = dropout(tf.keras.layers.Dense(self.filters)(tf.concat([embedding_context, cqa, embedding_context * cqa, embedding_context * qca], -1)), self.training)  #  B*CL*F

        # 1.4 Model Encoder
        model_encoder_blocks=[EncoderBlock(self.kernel_size2,self.filters,self.conv_layers2,self.heads,self.keep_prob) for _ in range(self.model_encoder_layers)]
        m0 = co_attention_output
        for model_encoder_block in model_encoder_blocks:
            m0 = model_encoder_block(m0 + position_repr,self.training,context_mask)
        m1 = m0
        for model_encoder_block in model_encoder_blocks:
            m1 = model_encoder_block(m1 + position_repr,self.training,context_mask)
        m2 = m1
        for model_encoder_block in model_encoder_blocks:
            m2 = model_encoder_block(m2 + position_repr,self.training,context_mask)
        norm_m0 = tf.contrib.layers.layer_norm(m0, begin_norm_axis=-1)
        norm_m1 = tf.contrib.layers.layer_norm(m1, begin_norm_axis=-1)
        norm_m2 = tf.contrib.layers.layer_norm(m2, begin_norm_axis=-1)
        # predict start

        start_logits = tf.keras.layers.Dense(1)(tf.concat([norm_m0, norm_m1], -1))[:, :, 0] + context_mask
        self.start_prob = tf.nn.softmax(start_logits,-1)
        self.start_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=start_logits,labels=tf.one_hot(self.answer_start,max_context_len)))

        # predict end
        end_logits = tf.keras.layers.Dense(1)(tf.concat([norm_m0, norm_m2], -1))[:, :, 0] + context_mask
        self.end_prob = tf.nn.softmax(end_logits,-1)
        self.end_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=end_logits, labels=tf.one_hot(self.answer_end,max_context_len)))

        self.loss = self.start_loss + self.end_loss
        self.global_step = tf.train.get_or_create_global_step()
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


        self.input_placeholder_dict = OrderedDict(input_dict)
        print(self.input_placeholder_dict)# = OrderedDict(input_dict)

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

    def compile(self, optimizer, initial_lr, clip_gradient=5.0):
        self.optimizer = optimizer(initial_lr)
        grads, vars = zip(*self.optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(grads, clip_gradient)
        opt_op = self.optimizer.apply_gradients(zip(gradients, vars), global_step=self.global_step)
        if self.ema_decay <= 0:
            self.train_op = opt_op
            return
        self.var_ema = tf.train.ExponentialMovingAverage(self.ema_decay)
        with tf.control_dependencies([opt_op]):
            self.train_op = self.var_ema.apply(tf.trainable_variables())
        self.restore_ema_variables = []
        self.ema_placeholders = []
        self.restore_cur_variables = []
        for var in tf.trainable_variables():
            self.restore_ema_variables.append(tf.assign(var, self.var_ema.average(var)))
            self.ema_placeholders.append(tf.placeholder(var.dtype, var.get_shape()))
            self.restore_cur_variables.append(tf.assign(var, self.ema_placeholders[-1]))

    def get_best_answer(self, output, instances,max_len=15):
        answer_list = []
        for i in range(len(output['start_prob'])):
            instance = instances[i]
            max_prob = 0.0
            start_position = 0
            end_position = 0
            d = deque()
            start_prob,end_prob = output['start_prob'][i],output['end_prob'][i]
            for idx in range(len(start_prob)):
                while len(d) > 0 and idx - d[0] >= max_len:
                    d.popleft()
                while len(d) > 0 and start_prob[d[-1]] <= start_prob[idx]:
                    d.pop()
                d.append(idx)
                if start_prob[d[0]] * end_prob[idx] > max_prob:
                    start_position = d[0]
                    end_position = idx
                    max_prob = start_prob[d[0]] * end_prob[idx]
            char_start_position = instance["context_token_spans"][start_position][0]
            char_end_position = instance["context_token_spans"][end_position][1]
            pred_answer = instance["context"][char_start_position:char_end_position]
            answer_list.append(pred_answer)
        return answer_list

    def _get_score(self, pred_list, evaluator):
        return evaluator.get_score(pred_list)

    def train_and_evaluate(self, train_generator, eval_generator, evaluator, epochs=1, eposides=1,
                           save_dir=None, summary_dir=None, save_summary_steps=10):
        if not self.initialized:
            self.session.run(tf.global_variables_initializer())

        Trainer._train_and_evaluate(self, train_generator, eval_generator, evaluator, epochs=epochs,
                                    eposides=eposides,
                                    save_dir=save_dir, summary_dir=summary_dir, save_summary_steps=save_summary_steps)

    def evaluate(self, batch_generator, evaluator):

        Trainer._evaluate(self, batch_generator, evaluator)

    def inference(self, batch_generator):
        Trainer._inference(self, batch_generator)