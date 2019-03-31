# coding: utf-8
from sogou_mrc.data.vocabulary import Vocabulary
from sogou_mrc.data.batch_generator import BatchGenerator
from sogou_mrc.dataset.squadv2 import SquadV2Reader, SquadV2Evaluator
from sogou_mrc.model.bidafplus_squad2 import BiDAFPlus
import tensorflow as tf
import logging
import json
import time
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "0"

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

train_file = "train-v2.0.json"
dev_file = "dev-v2.0.json"

t0 = time.time()
reader = SquadV2Reader()
train_data = reader.read(train_file)
eval_data = reader.read(dev_file)
evaluator = SquadV2Evaluator(dev_file)
cost = time.time() - t0
logging.info("seg cost=%.3f" % cost)

t0 = time.time()
vocab = Vocabulary(do_lowercase=True)
vocab.build_vocab(train_data + eval_data, min_word_count=3, min_char_count=10)
word_embedding = vocab.make_word_embedding("glove.840B.300d.txt", init_scale=0.05)
cost = time.time() - t0
logging.info("make vocab cost=%.3f" % cost)

train_batch_generator = BatchGenerator(vocab, train_data, batch_size=16, training=True, additional_fields=["abstractive_answer_mask"])
eval_batch_generator = BatchGenerator(vocab, eval_data, batch_size=16, training=False, additional_fields=["abstractive_answer_mask"])

use_elmo=True
save_path="squad2_elmo"

if use_elmo:
    model = BiDAFPlus(vocab, pretrained_word_embedding=word_embedding, abstractive_answer=[""], use_elmo=True, elmo_local_path="path_to_elmo")
else:
    model = BiDAFPlus(vocab, pretrained_word_embedding=word_embedding, abstractive_answer=[""])

model.compile(tf.train.AdadeltaOptimizer, 1.0)
model.train_and_evaluate(train_batch_generator, eval_batch_generator, evaluator, epochs=40, eposides=2, save_dir=save_path)
