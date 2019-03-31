# coding:utf-8
import tensorflow as tf
from sogou_mrc.nn.layers import Layer


class DotProduct(Layer):
    def __init__(self, scale=False, name="dot_product"):
        super(DotProduct, self).__init__(name)
        self.scale = scale

    def __call__(self, t0, t1):
        dots = tf.matmul(t0, t1, transpose_b=True)
        if self.scale:
            last_dims = t0.shape.as_list()[-1]
            dots = dots / tf.sqrt(last_dims)
        return dots


class ProjectedDotProduct(Layer):
    def __init__(self, hidden_units, activation=None, reuse_weight=False, name="projected_dot_product"):
        super(ProjectedDotProduct, self).__init__(name)
        self.reuse_weight = reuse_weight
        self.projecting_layer = tf.keras.layers.Dense(hidden_units, activation=activation,
                                                      use_bias=False)
        if not reuse_weight:
            self.projecting_layer2 = tf.keras.layers.Dense(hidden_units, activation=activation,
                                                           use_bias=False)

    def __call__(self, t0, t1):
        t0 = self.projecting_layer(t0)
        if self.reuse_weight:
            t1 = self.projecting_layer(t1)
        else:
            t1 = self.projecting_layer2(t1)

        return tf.matmul(t0, t1, transpose_b=True)


class BiLinear(Layer):
    def __init__(self, name="bi_linear"):
        super(BiLinear, self).__init__(name)
        self.projecting_layer = None

    def __call__(self, t0, t1):
        hidden_units = t0.shape.as_list()[-1]
        if self.projecting_layer is None:
            self.projecting_layer = tf.keras.layers.Dense(hidden_units, activation=None,
                                                          use_bias=False)

        t0 = self.projecting_layer(t0)
        return tf.matmul(t0, t1, transpose_b=True)


class TriLinear(Layer):
    def __init__(self, name="tri_linear", bias=False):
        super(TriLinear, self).__init__(name)
        self.projecting_layers = [tf.keras.layers.Dense(1, activation=None, use_bias=False) for _ in range(2)]
        self.dot_w = None
        self.bias = bias

    def __call__(self, t0, t1):
        t0_score = tf.squeeze(self.projecting_layers[0](t0), axis=-1)
        t1_score = tf.squeeze(self.projecting_layers[1](t1), axis=-1)

        if self.dot_w is None:
            hidden_units = t0.shape.as_list()[-1]
            with tf.variable_scope(self.name):
                self.dot_w = tf.get_variable("dot_w", [hidden_units])

        t0_dot_w = t0 * tf.expand_dims(tf.expand_dims(self.dot_w, axis=0), axis=0)
        t0_t1_score = tf.matmul(t0_dot_w, t1, transpose_b=True)

        out = t0_t1_score + tf.expand_dims(t0_score, axis=2) + tf.expand_dims(t1_score, axis=1)
        if self.bias:
            with tf.variable_scope(self.name):
                bias = tf.get_variable("bias", shape=(), dtype=tf.float32)
            out += bias
        return out


class MLP(Layer):
    def __init__(self, hidden_units, activation=tf.nn.tanh, name="mlp"):
        super(MLP, self).__init__(name)
        self.activation = activation
        self.projecting_layers = [tf.keras.layers.Dense(hidden_units, activation=None) for _ in range(2)]
        self.score_layer = tf.keras.layers.Dense(1, activation=None, use_bias=False)

    def __call__(self, t0, t1):
        t0 = self.projecting_layers[0](t0)
        t1 = self.projecting_layers[1](t1)
        t0_t1 = tf.expand_dims(t0, axis=2) + tf.expand_dims(t1, axis=1)
        return tf.squeeze(self.score_layer(self.activation(t0_t1)), axis=-1)


class SymmetricProject(Layer):
    def __init__(self, hidden_units, reuse_weight=True, activation=tf.nn.relu, name='symmetric_nolinear'):
        super(SymmetricProject, self).__init__(name)
        self.reuse_weight = reuse_weight
        self.hidden_units = hidden_units
        with tf.variable_scope(self.name):
            diagonal = tf.get_variable('diagonal_matrix', shape=[self.hidden_units],initializer=tf.ones_initializer, dtype=tf.float32)
        self.diagonal_matrix = tf.diag(diagonal)
        self.projecting_layer = tf.keras.layers.Dense(hidden_units, activation=activation,
                                                      use_bias=False)
        if not reuse_weight:
            self.projecting_layer2 = tf.keras.layers.Dense(hidden_units, activation=activation, use_bias=False)

    def __call__(self, t0, t1):
        trans_t0 = self.projecting_layer(t0)
        trans_t1 = self.projecting_layer(t1)
        return tf.matmul(tf.tensordot(trans_t0,self.diagonal_matrix,[[2],[0]]),trans_t1,transpose_b=True)
