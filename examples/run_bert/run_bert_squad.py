# coding: utf-8
import sys
from sogou_mrc.data.vocabulary import Vocabulary
from sogou_mrc.dataset.squad import SquadReader, SquadEvaluator
from sogou_mrc.model.bert import BertBaseline
from sogou_mrc.libraries.BertWrapper import BertDataHelper
import logging
import random
random.seed(1234)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
data_folder = ''
train_file = data_folder + "train-v1.1.json"
dev_file = data_folder + "dev-v1.1.json"
reader = SquadReader()
train_data = reader.read(train_file)
random.shuffle(train_data)

eval_data = reader.read(dev_file)
evaluator = SquadEvaluator(dev_file)
vocab = Vocabulary(do_lowercase=True)
bert_dir = ''
bert_data_helper = BertDataHelper(bert_dir)
vocab.build_vocab(train_data + eval_data, min_word_count=3, min_char_count=10)

#covert data to bert format

train_data = bert_data_helper.convert(train_data,data='squad')
eval_data = bert_data_helper.convert(eval_data,data='squad')

from sogou_mrc.data.batch_generator import BatchGenerator
train_batch_generator = BatchGenerator(vocab,train_data,training=True,batch_size=12,additional_fields=['input_ids','segment_ids','input_mask','start_position','end_position'])
eval_batch_generator = BatchGenerator(vocab,eval_data,training=False,batch_size=12,additional_fields=['input_ids','segment_ids','input_mask','start_position','end_position'])
model = BertBaseline(bert_dir=bert_dir,version_2_with_negative=False)
warmup_proportion = 0.1
num_train_steps = int(
        len(train_data) / 12 * 2)
num_warmup_steps = int(num_train_steps * warmup_proportion)

model.compile(3e-5,num_train_steps=num_train_steps,num_warmup_steps=num_warmup_steps)
model.train_and_evaluate(train_batch_generator, eval_batch_generator,evaluator, epochs=2, eposides=1)
