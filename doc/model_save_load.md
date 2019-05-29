### example shows how to save and load model
save_dir should be passed to `train_and_evaluate`, best models will be saved in the path save_dir+'/best_weights'.
```python
# coding: utf-8
from sogou_mrc.data.vocabulary import Vocabulary
from sogou_mrc.dataset.squad import SquadReader, SquadEvaluator
from sogou_mrc.model.bidaf import BiDAF
import tensorflow as tf
import logging
from sogou_mrc.data.batch_generator import BatchGenerator

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

vocab = Vocabulary()
vocab.build_vocab(train_data + eval_data, min_word_count=3, min_char_count=10)
word_embedding = vocab.make_word_embedding(embedding_folder + "glove.6B.100d.txt")
#save vocab 
vocab_save_path='xx/vocab.json'
vocab.save(vocab_save_path)

train_batch_generator = BatchGenerator(vocab, train_data, batch_size=60, training=True)
eval_batch_generator = BatchGenerator(vocab, eval_data, batch_size=60)

#train and save checkpoint in save_dir 
save_dir='' #define save_dir path 
model = BiDAF(vocab, pretrained_word_embedding=word_embedding)
model.compile(tf.train.AdamOptimizer, 0.001)
model.train_and_evaluate(train_batch_generator, eval_batch_generator, evaluator, epochs=15, eposides=2, save_dir=save_dir)

````
After the model was saved, best checkpoint can be restored by calling load method.
```python

 coding: utf-8
from sogou_mrc.data.vocabulary import Vocabulary
from sogou_mrc.dataset.squad import SquadReader, SquadEvaluator
from sogou_mrc.model.bidaf import BiDAF
import tensorflow as tf
import logging
from sogou_mrc.data.batch_generator import BatchGenerator

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

data_folder = ''
dev_file = data_folder + "dev-v1.1.json"

reader = SquadReader()
eval_data = reader.read(dev_file)
evaluator = SquadEvaluator(dev_file)

vocab = Vocabulary()
vocab_save_path='xx/vocab.json'
vocab.load(vocab_save_path) # load vocab from save path 

test_batch_generator = BatchGenerator(vocab, eval_data, batch_size=60)

save_dir='xxx'+'/best_weights'
model = BiDAF(vocab)
model.load(save_dir)
model.inference(test_batch_generator) # inference on test data
```
