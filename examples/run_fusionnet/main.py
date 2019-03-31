# coding: utf-8
from sogou_mrc.data.vocabulary import Vocabulary
from sogou_mrc.dataset.squad import SquadReader, SquadEvaluator
from sogou_mrc.model.fusionnet import FusionNet
import tensorflow as tf
import logging
from sogou_mrc.data.batch_generator import BatchGenerator
from sogou_mrc.utils.feature_extractor import FeatureExtractor
tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
data_folder = ''
embedding_folder = ''
train_file = data_folder + "train-v1.1.json"
dev_file = data_folder + "dev-v1.1.json"

reader = SquadReader()
train_data = reader.read(train_file)
eval_data = reader.read(dev_file)
evaluator = SquadEvaluator(dev_file)

vocab = Vocabulary(do_lowercase=False)
vocab.build_vocab(train_data + eval_data, min_word_count=3, min_char_count=10)
word_embedding = vocab.make_word_embedding(embedding_folder + "glove.840B.300d.txt")


feature_transformer = FeatureExtractor(features=['match_lemma', 'match_lower', 'pos', 'ner','context_tf'],
                                       build_vocab_feature_names=set(['pos', 'ner']),
                                       word_counter = vocab.get_word_counter()
                                       )
train_data = feature_transformer.fit_transform(dataset=train_data)
eval_data = feature_transformer.transform(dataset=eval_data)

train_batch_generator = BatchGenerator(vocab, train_data, batch_size=32,
                                       training=True,
                                       additional_fields=feature_transformer.features,
                                       feature_vocab=feature_transformer.vocab)

eval_batch_generator = BatchGenerator(vocab, eval_data, batch_size=32,
                                      additional_fields=feature_transformer.features,
                                      feature_vocab=feature_transformer.vocab)

model = FusionNet(vocab, word_embedding, features=feature_transformer.features, feature_vocab=feature_transformer.vocab)
# adamax optimizer
model.compile()
model.train_and_evaluate(train_batch_generator, eval_batch_generator, evaluator, epochs=50, eposides=2)
model.evaluate(eval_batch_generator, evaluator)
