# coding:utf-8

from sogou_mrc.model.base_model import BaseModel
from sogou_mrc.nn.recurrent import CudnnBiLSTM,CudnnBiGRU,CudnnGRU
from sogou_mrc.nn.layers import VariationalDropout, Dropout, Embedding, MultiLayerRNN, MultiHeadAttention
from collections import OrderedDict,deque
import tensorflow as tf


class RNET(BaseModel):
    def __init__(self, vocab, pretrained_word_embedding=None,word_embedding_size=300,char_embedding_size=100,hidden_size=75,
                 dropout_keep_prob=0.8, word_embedding_trainable=False):
        super(RNET, self).__init__(vocab)
        self.doc_rnn_layers = 1
        self.question_rnn_layers = 1
        self.heads = 3
        self.hidden_size = hidden_size
        self.word_embedding_size =word_embedding_size
        self.keep_prob = dropout_keep_prob
        self.pretrained_word_embedding = pretrained_word_embedding
        self.char_embedding_size = char_embedding_size
        self.word_embedding_trainable = word_embedding_trainable
        self._build_graph()
        self.initialized = False

    def _build_graph(self):
        # build input

        self.context_word = tf.placeholder(tf.int32, [None, None])
        self.context_char = tf.placeholder(tf.int32, [None, None, None])
        self.context_len = tf.placeholder(tf.int32, [None])
        self.context_word_len = tf.placeholder(tf.int32, [None, None])
        self.question_word = tf.placeholder(tf.int32, [None, None])
        self.question_char = tf.placeholder(tf.int32, [None, None, None])
        self.question_len = tf.placeholder(tf.int32, [None])
        self.question_word_len = tf.placeholder(tf.int32, [None, None])
        self.answer_start = tf.placeholder(tf.int32, [None])
        self.answer_end = tf.placeholder(tf.int32, [None])
        self.training = tf.placeholder(tf.bool, [])

        max_context_len = tf.shape(self.context_word)[1]
        max_context_word_len = tf.shape(self.context_char)[2]
        max_question_len = tf.shape(self.question_word)[1]
        max_question_word_len = tf.shape(self.question_char)[2]
        context_mask = (tf.sequence_mask(self.context_len, max_context_len, dtype=tf.float32)-1)*100
        question_mask = (tf.sequence_mask(self.question_len, max_question_len, dtype=tf.float32)-1) * 100
        # 1. Word encoding
        word_embedding = Embedding(pretrained_embedding=self.pretrained_word_embedding,
                                   embedding_shape=(len(self.vocab.get_word_vocab()) + 1, self.word_embedding_size),
                                   trainable=self.word_embedding_trainable)
        char_embedding = Embedding(embedding_shape=(len(self.vocab.get_char_vocab()) + 1, self.char_embedding_size),
                                   trainable=True, init_scale=0.2)
        dropout = VariationalDropout(self.keep_prob)
        context_word_embedding = word_embedding(self.context_word)  # B*CL*WD
        context_char_embedding = dropout(tf.reshape(char_embedding(self.context_char),[-1, max_context_word_len, self.char_embedding_size]),self.training)  # (B*CL)*WL*CD
        question_word_embedding = word_embedding(self.question_word)  # B*QL*WD
        question_char_embedding = dropout(tf.reshape(char_embedding(self.question_char),[-1, max_question_word_len, self.char_embedding_size]),self.training)  # (B*QL)*WL*CD

        char_forward_rnn = tf.keras.layers.CuDNNGRU(self.hidden_size)
        char_backward_rnn = tf.keras.layers.CuDNNGRU(self.hidden_size, go_backwards=True)
        context_char_forward_final_states = char_forward_rnn(context_char_embedding)  # (B*CL)*H
        context_char_backward_final_states = char_backward_rnn(context_char_embedding)  # (B*CL)*H
        context_char_final_states = tf.reshape(tf.concat([context_char_forward_final_states, context_char_backward_final_states], -1),[-1, max_context_len,self.hidden_size * 2])
        context_repr = tf.concat([context_word_embedding, context_char_final_states], -1) # B*CL*(WD+H)

        question_char_forword_final_states = char_forward_rnn(question_char_embedding)  # (B*CL)*H
        question_char_backword_final_states = char_backward_rnn(question_char_embedding)  # (B*CL)*H
        question_char_final_states = tf.reshape(tf.concat([question_char_forword_final_states, question_char_backword_final_states], -1),[-1, max_question_len,self.hidden_size * 2])
        question_repr = tf.concat([question_word_embedding, question_char_final_states],-1)  # B*QL*(WD+H)

        # 1.2 Encoder
        context_rnn = [CudnnBiGRU(self.hidden_size) for _ in range(self.doc_rnn_layers)]
        encoder_multi_bigru = MultiLayerRNN(context_rnn, concat_layer_out=True,input_keep_prob=self.keep_prob)
        encoder_context = dropout(encoder_multi_bigru(context_repr, self.context_len,self.training),self.training)  # B*CL*(H*2)
        encoder_question = dropout(encoder_multi_bigru(question_repr, self.question_len,self.training),self.training)  # B*QL*(H*2)

        # 1.3 co-attention
        co_attention_context = tf.expand_dims(tf.keras.layers.Dense(self.hidden_size)(encoder_context),2) # B*CL*1*H
        co_attention_question = tf.expand_dims(tf.keras.layers.Dense(self.hidden_size)(encoder_question), 1) # B*1*QL*H
        co_attention_score = tf.keras.layers.Dense(1)(tf.nn.tanh(co_attention_context+co_attention_question))[:,:,:,0]+tf.expand_dims(question_mask,1) # B*CL*QL
        co_attention_similarity = tf.nn.softmax(co_attention_score,-1) # B*CL*QL
        co_attention_rnn_input = tf.concat([encoder_context,tf.matmul(co_attention_similarity,encoder_question)],-1) # B*CL*(H*4)
        co_attention_rnn_input = co_attention_rnn_input*tf.keras.layers.Dense(self.hidden_size*4,activation=tf.nn.sigmoid)(co_attention_rnn_input)
        co_attention_rnn = CudnnGRU(self.hidden_size)
        co_attention_output = dropout(co_attention_rnn(co_attention_rnn_input,self.context_len)[0],self.training)  # B*CL*(H*2)

        # 1.4 self-attention
        multi_head_attention = MultiHeadAttention(self.heads,self.hidden_size,False)
        self_attention_repr = dropout(multi_head_attention(co_attention_output,co_attention_output,co_attention_output,context_mask),self.training)
        self_attention_rnn_input = tf.concat([co_attention_output,self_attention_repr],-1) # B*CL*(H*2)
        self_attention_rnn_input = self_attention_rnn_input*tf.keras.layers.Dense(self.hidden_size*2,activation=tf.nn.sigmoid)(self_attention_rnn_input)
        self_attention_rnn = CudnnBiGRU(self.hidden_size)
        self_attention_output = dropout(self_attention_rnn(self_attention_rnn_input, self.context_len)[0], self.training)  # B*CL*(H*2)

        # predict_start
        start_question_score = tf.keras.layers.Dense(1)(tf.keras.layers.Dense(self.hidden_size, activation=tf.nn.tanh)(encoder_question)) + tf.expand_dims(question_mask,-1) # B*QL*1
        start_question_similarity = tf.nn.softmax(start_question_score,1) # B*QL*1
        start_question_repr = tf.matmul(start_question_similarity, encoder_question, transpose_a=True)[:,0] # B*(H*2)
        start_logits = tf.keras.layers.Dense(1)(tf.nn.tanh(tf.keras.layers.Dense(self.hidden_size)(self_attention_output) + tf.expand_dims(tf.keras.layers.Dense(self.hidden_size)(start_question_repr), 1)))[:, :, 0] + context_mask # B*CL
        self.start_prob = tf.nn.softmax(start_logits, -1)  # B*CL
        self.start_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=start_logits,labels=self.answer_start))

        start_repr = tf.matmul(tf.expand_dims(self.start_prob, 1),self_attention_output) #B*1*(H*2)
        start_output = Dropout(self.keep_prob)(tf.keras.layers.CuDNNGRU(self.hidden_size * 2)(start_repr,start_question_repr),self.training) #B*(H*2)

        # predict_end
        end_logits = tf.keras.layers.Dense(1)(tf.nn.tanh(tf.keras.layers.Dense(self.hidden_size)(self_attention_output) + tf.expand_dims(tf.keras.layers.Dense(self.hidden_size)(start_output), 1)))[:, :, 0] + context_mask # B*CL
        self.end_prob = tf.nn.softmax(end_logits, -1)  # B*CL
        self.end_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_logits, labels=self.answer_end))

        self.loss = self.start_loss + self.end_loss
        self.global_step = tf.train.get_or_create_global_step()
        input_dict = {
            "context_word": self.context_word,
            "context_char": self.context_char,
            "context_len": self.context_len,
            "context_word_len": self.context_word_len,
            "question_word": self.question_word,
            "question_char": self.question_char,
            "question_len": self.question_len,
            "question_word_len": self.question_word_len,
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
        self.train_op = self.optimizer.apply_gradients(zip(gradients, vars))

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
