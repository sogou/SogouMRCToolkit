#coding:utf-8

# coding: utf-8
from sogou_mrc.data.vocabulary import Vocabulary
from sogou_mrc.dataset.squadv2 import SquadV2Reader, SquadV2Evaluator
from sogou_mrc.model.bidaf import BiDAF
import tensorflow as tf
import logging
from sogou_mrc.data.batch_generator import BatchGenerator

tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

data_folder = ''
embedding_folder = '/search/odin/jdwu/glove/'
train_file = data_folder + "train-v2.0.json"
dev_file = data_folder + "dev-v2.0.json"

reader = SquadV2Reader()
train_data = reader.read(train_file)
eval_data = reader.read(dev_file)
evaluator = SquadV2Evaluator(dev_file)

vocab = Vocabulary()
vocab.build_vocab(train_data + eval_data, min_word_count=3, min_char_count=10)
word_embedding = vocab.make_word_embedding(embedding_folder + "glove.6B.100d.txt")

train_batch_generator = BatchGenerator(vocab, train_data, batch_size=60, training=True,additional_fields=['is_impossible'])
eval_batch_generator = BatchGenerator(vocab, eval_data, batch_size=60,additional_fields=['is_impossible'])
model = BiDAF(vocab, pretrained_word_embedding=word_embedding,enable_na_answer=True)
model.compile(tf.train.AdamOptimizer, 0.001)
model.train_and_evaluate(train_batch_generator, eval_batch_generator, evaluator, epochs=15, eposides=2)
