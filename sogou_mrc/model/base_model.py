# coding:utf-8
import tensorflow as tf
import logging
import numpy as np
import os
from collections import OrderedDict, defaultdict
from sogou_mrc.train.trainer import Trainer


class BaseModel(object):
    def __init__(self, vocab=None):
        self.vocab = vocab

        sess_conf = tf.ConfigProto()
        sess_conf.gpu_options.allow_growth = True
        self.session = tf.Session(config=sess_conf)
        self.initialized = False
        self.ema_decay = 0

    def __del__(self):
        self.session.close()

    def load(self, path, var_list=None):
        # var_list = None returns the list of all saveable variables
        logging.info('Loading model from %s' % path)
        saver = tf.train.Saver(var_list)
        checkpoint_path = tf.train.latest_checkpoint(path)
        saver.restore(self.session, save_path=checkpoint_path)
        self.initialized = True

    def save(self, path, global_step=None, var_list=None):
        saver = tf.train.Saver(var_list)
        saver.save(self.session, path, global_step=global_step)

    def _build_graph(self):
        raise NotImplementedError

    def compile(self, *input):
        raise NotImplementedError

    def get_best_answer(self, *input):
        raise NotImplementedError

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
