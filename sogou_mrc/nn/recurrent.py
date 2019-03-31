# coding: utf-8
import tensorflow as tf
from sogou_mrc.nn.layers import Layer


class BaseBiRNN(Layer):
    def __init__(self, name="base_BiRNN"):
        super(BaseBiRNN, self).__init__(name)
        self.fw_cell = None
        self.bw_cell = None

    def __call__(self, seq, seq_len):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, seq, seq_len, dtype=tf.float32)
        if isinstance(states[0], tuple):
            return tf.concat(outputs, axis=-1), tf.concat([s.h for s in states], axis=-1)
        else:
            return tf.concat(outputs, axis=-1), tf.concat(states, axis=-1)


class BiLSTM(BaseBiRNN):
    def __init__(self, hidden_units, name="BiLSTM"):
        super(BiLSTM, self).__init__(name)
        self.fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_units,name=name+'_fw_cell')
        self.bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_units,name=name+'_bw_cell')


class BiGRU(BaseBiRNN):
    def __init__(self, hidden_units, name="BiGRU"):
        super(BiGRU, self).__init__(name)
        self.fw_cell = tf.nn.rnn_cell.GRUCell(hidden_units,name=name+'_fw_cell')
        self.bw_cell = tf.nn.rnn_cell.GRUCell(hidden_units,name=name+'_bw_cell')


class BaseCudnnBiRNN(Layer):
    def __init__(self, name="base_cudnn_BiRNN"):
        super(BaseCudnnBiRNN, self).__init__(name)
        self.fw_layer = None
        self.bw_layer = None

    def __call__(self, seq, seq_len):
        fw = self.fw_layer(seq)
        bw = self.bw_layer(tf.reverse_sequence(seq, seq_len, 1, 0))
        bw = tf.reverse_sequence(bw, seq_len, 1, 0)
        return tf.concat((fw, bw), axis=-1), None


class CudnnBiLSTM(BaseCudnnBiRNN):
    """
    Final states is not available currently.
    """
    def __init__(self, hidden_units, name="cudnn_BiLSTM"):
        super(CudnnBiLSTM, self).__init__(name)
        self.fw_layer = tf.keras.layers.CuDNNLSTM(hidden_units, return_sequences=True)
        self.bw_layer = tf.keras.layers.CuDNNLSTM(hidden_units, return_sequences=True)


class CudnnBiGRU(BaseCudnnBiRNN):
    """
    Final states is not available currently.
    """
    def __init__(self, hidden_units, name="cudnn_BiGRU"):
        super(CudnnBiGRU, self).__init__(name)
        self.fw_layer = tf.keras.layers.CuDNNGRU(hidden_units, return_sequences=True)
        self.bw_layer = tf.keras.layers.CuDNNGRU(hidden_units, return_sequences=True)

class BaseCudnnRNN(Layer):
    def __init__(self,name="base_cudnn_RNN"):
        super(BaseCudnnRNN, self).__init__(name)
        self.fw_layer = None

    def __call__(self, seq, seq_len):
        fw = self.fw_layer(seq)
        return fw, None

class CudnnGRU(BaseCudnnRNN):
    """
    Final states is not available currently.
    """
    def __init__(self, hidden_units, name="cudnn_GRU"):
        super(CudnnGRU, self).__init__(name)
        self.fw_layer = tf.keras.layers.CuDNNGRU(hidden_units, return_sequences=True)