# coding: utf-8
from sogou_mrc.data.vocabulary import Vocabulary
from sogou_mrc.dataset.squad import SquadReader, SquadEvaluator
from sogou_mrc.dataset.cmrc import CMRCReader,CMRCEvaluator
from sogou_mrc.model.bidaf import BiDAF
import tensorflow as tf
import logging
from sogou_mrc.data.batch_generator import BatchGenerator

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
data_folder = ''
embedding_folder = ''
train_file = data_folder+"cmrc2018_train.json"
dev_file = data_folder+"cmrc2018_dev.json"

reader = CMRCReader()
train_data = reader.read(train_file)
eval_data = reader.read(dev_file)
evaluator = CMRCEvaluator(dev_file)

vocab = Vocabulary(do_lowercase=False)
vocab.build_vocab(train_data + eval_data, min_word_count=3, min_char_count=10)
word_embedding = vocab.make_word_embedding(embedding_folder )
train_batch_generator = BatchGenerator(vocab, train_data, batch_size=32, training=True)
eval_batch_generator = BatchGenerator(vocab, eval_data, batch_size=60)

model = BiDAF(vocab, pretrained_word_embedding=word_embedding,word_embedding_size=300)
model.compile(tf.train.AdamOptimizer, 0.001)
model.train_and_evaluate(train_batch_generator, eval_batch_generator, evaluator, epochs=50, eposides=2)
