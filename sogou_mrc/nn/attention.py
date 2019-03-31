# coding:utf-8

import tensorflow as tf
from sogou_mrc.nn.layers import Layer

VERY_NEGATIVE_NUMBER = -1e29

class BiAttention(Layer):
    """ Bi-Directonal Attention from https://arxiv.org/abs/1611.01603 """

    def __init__(self, similarity_function, name="bi_attention"):
        super(BiAttention, self).__init__(name)
        self.similarity_function = similarity_function

    def __call__(self, query, memory, query_len, memory_len):
        sim_mat = self.similarity_function(query, memory)
        mask = tf.expand_dims(tf.sequence_mask(query_len, tf.shape(query)[1], dtype=tf.float32), axis=2) * \
               tf.expand_dims(tf.sequence_mask(memory_len, tf.shape(memory)[1], dtype=tf.float32), axis=1)
        sim_mat = sim_mat + (1. - mask) * tf.float32.min

        # Context-to-query Attention in the paper
        query_memory_prob = tf.nn.softmax(sim_mat)
        query_memory_attention = tf.matmul(query_memory_prob, memory)

        # Query-to-context Attention in the paper
        memory_query_prob = tf.nn.softmax(tf.reduce_max(sim_mat, axis=-1))
        memory_query_attention = tf.matmul(tf.expand_dims(memory_query_prob, axis=1), query)
        memory_query_attention = tf.tile(memory_query_attention, [1, tf.shape(query)[1], 1])

        return query_memory_attention, memory_query_attention


class UniAttention(Layer):
    """ Commonly used Uni-Directional Attention"""

    def __init__(self, similarity_function, name="uni_attention"):
        super(UniAttention, self).__init__(name)
        self.similarity_function = similarity_function

    def __call__(self, query, key, key_len, value=None):
        # If value is not given, key will be treated as value
        sim_mat = self.similarity_function(query, key)
        mask = tf.expand_dims(tf.sequence_mask(key_len, tf.shape(key)[1], dtype=tf.float32), axis=1)
        sim_mat = sim_mat + (1. - mask) * tf.float32.min

        sim_prob = tf.nn.softmax(sim_mat)
        if value is not None:
            return tf.matmul(sim_prob, value)
        else:
            return tf.matmul(sim_prob, key)


class SelfAttention(Layer):
    def __init__(self, similarity_function, name="self_attention"):
        super(SelfAttention, self).__init__(name)
        self.similarity_function = similarity_function

    def __call__(self, query, query_len):
        sim_mat = self.similarity_function(query, query)
        sim_mat += tf.expand_dims(tf.eye(tf.shape(query)[1]) * VERY_NEGATIVE_NUMBER, 0)
        mask = tf.expand_dims(tf.sequence_mask(query_len, tf.shape(query)[1], dtype=tf.float32), axis=1)
        sim_mat = sim_mat + (1. - mask) * VERY_NEGATIVE_NUMBER 
        bias = tf.exp(tf.get_variable("no-alignment-bias", initializer=tf.constant(-1.0, dtype=tf.float32)))
        sim_mat = tf.exp(sim_mat)
        sim_prob = sim_mat / (tf.reduce_sum(sim_mat, axis=2, keep_dims=True) + bias)

        return tf.matmul(sim_prob, query)



class SelfAttn(Layer):
    def __init__(self, name='self_attn', use_bias=False):
        super(SelfAttn, self).__init__(name)
        self.project = tf.keras.layers.Dense(1, use_bias=use_bias)

    def __call__(self, input, input_len):
        mask = tf.sequence_mask(input_len, tf.shape(input)[1], dtype=tf.float32)
        sim = tf.squeeze(self.project(input), axis=-1)
        sim = sim + (1.0 - mask) * tf.float32.min
        # scores
        sim_mat_score = tf.nn.softmax(sim)
        return tf.reduce_sum(input * tf.expand_dims(sim_mat_score, axis=-1), axis=1)
