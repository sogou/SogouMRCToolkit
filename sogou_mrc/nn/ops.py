import tensorflow as tf


def dropout(x, keep_prob, training, noise_shape=None):
    if keep_prob >= 1.0:
        return x
    return tf.cond(training, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape), lambda: x)


def weighted_sum(seq, prob):
    return tf.reduce_sum(seq * tf.expand_dims(prob, axis=2), axis=1)


def masked_softmax(logits, mask):
    if len(logits.shape.as_list()) != len(mask.shape.as_list()):
        mask = tf.sequence_mask(mask, tf.shape(logits)[1], dtype=tf.float32)

    return tf.nn.softmax(logits + (1.0 - mask) * tf.float32.min)


def mask_logits(logits, mask):
    if len(logits.shape.as_list()) != len(mask.shape.as_list()):
        mask = tf.sequence_mask(mask, tf.shape(logits)[1], dtype=tf.float32)

    return logits + (1.0 - mask) * tf.float32.min

def add_seq_mask(inputs, seq_len, mode='mul', max_len=None):
    mask = tf.expand_dims(tf.cast(tf.sequence_mask(seq_len, maxlen=max_len), tf.float32), 2)
    if mode == 'mul':
        return inputs * mask
    if mode == 'add':
        mask = (1 - mask) * tf.float32.min 
        return inputs + mask
