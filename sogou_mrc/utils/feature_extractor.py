# coding:utf-8
import spacy
import numpy as np
import logging
import json
from stanfordnlp.server.client import CoreNLPClient


class FeatureExtractor(object):

    def __init__(self, features=['pos', 'match_lemma', 'match_lower', 'ner', 'context_tf'],
                 build_vocab_feature_names=None, word_counter=None, language='en'):
        self.spacy_feature_extractor = SpacyFeatureExtractor() if language == 'en' else StanfordFeatureExtractor(language=language)
        self.vocab = {}
        self.vocab_set = {}
        self.features = features
        self.word_counter = word_counter
        self.build_vocab_feature_names = set() if build_vocab_feature_names is None else build_vocab_feature_names
        for feature in features:
            if feature in self.build_vocab_feature_names:
                self.vocab[feature] = ['<PAD>']
                self.vocab_set[feature] = set()

    def fit_transform(self, dataset):
        new_instances = [instance for instance in self._transform(dataset, is_training=True)]
        for feature in self.build_vocab_feature_names:
            self.vocab[feature].extend(self.vocab_set[feature])
        return new_instances

    def transform(self, dataset):
        new_instances = [instance for instance in self._transform(dataset)]
        return new_instances

    def _transform(self, dataset, is_training=False):
        for instance in dataset:
            features = self.spacy_feature_extractor.get_feature(instance['context'])
            question_feature = self.spacy_feature_extractor.get_feature(instance['question'])
            if 'match_lemma' in self.features:
                instance['match_lemma'] = np.array(
                    [w in set(question_feature['lemma_feature']) for w in features['lemma_feature']]).astype(int).tolist()
            if 'match_lower' in self.features:
                instance['match_lower'] = np.array([w.lower() in set([w.lower() for w in instance['question_tokens']])
                                                    for w in instance['context_tokens']]).astype(np.int).tolist()
            if 'context_tf' in self.features:
                instance['context_tf'] = [self.word_counter[w] / len(self.word_counter) for w in
                                          instance['context_tokens']]
            if len(self.build_vocab_feature_names) > 0:
                if is_training:
                    for feature in self.build_vocab_feature_names:
                        if feature in self.features and feature in features:
                            for tag in features[feature]:
                                self.vocab_set[feature].add(tag)
                for feature in self.build_vocab_feature_names:
                    if feature not in features:
                        print('feature not implemented')
                    else:
                        instance[feature] = features[feature]
            yield instance

    def save(self, file_path):
        logging.info("Saving vocabulary at {}".format(file_path))
        with open(file_path, "w") as f:
            json.dump({'vocab': self.vocab, 'features': self.features, 'build_vocab': list(self.build_vocab_feature_names),
                       'word_counter': self.word_counter}, f, indent=4)

    def load(self, file_path):
        logging.info("Loading vocabulary at {}".format(file_path))
        with open(file_path) as f:
            vocab_data = json.load(f)
            self.vocab = vocab_data['vocab']
            self.features = vocab_data['features']
            self.build_vocab_feature_names = set(vocab_data['build_vocab'])
            self.word_counter = vocab_data['word_counter']


class SpacyFeatureExtractor(object):
    def __init__(self):
        self.nlp = spacy.load("en", parse=False)

    def get_feature(self, doc):
        doc = self.nlp(doc)
        lemma_feature = [w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in doc]
        tags = [w.tag_ for w in doc]
        ents = [w.ent_type_ for w in doc]
        context_lower = [w.text.lower() for w in doc]
        return {'lemma_feature': lemma_feature, 'pos': tags, 'context_lower': context_lower, 'ner': ents}


class StanfordFeatureExtractor(object):
    def __init__(self, language='zh', annotators='ssplit tokenize ner pos lemma', timeout=30000, memory="4G"):
        if language == 'zh':
            CHINESE_PROPERTIES = {
                "tokenize.language": "zh",
                "segment.model": "edu/stanford/nlp/models/segmenter/chinese/ctb.gz",
                "segment.sighanCorporaDict": "edu/stanford/nlp/models/segmenter/chinese",
                "segment.serDictionary": "edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz",
                "segment.sighanPostProcessing": "true",
                "ssplit.boundaryTokenRegex": "[.。]|[!?！？]+",
            }
            if 'pos' in annotators:
                CHINESE_PROPERTIES[
                    'pos.model'] = "edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger"
            if 'ner' in annotators:
                CHINESE_PROPERTIES['ner.language'] = 'chinese'
                CHINESE_PROPERTIES['ner.model'] = 'edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz'
                CHINESE_PROPERTIES['ner.applyNumericClassifiers'] = 'true'
                CHINESE_PROPERTIES['ner.useSUTime'] = 'false'
        else:
            CHINESE_PROPERTIES = {}
        self.client = CoreNLPClient(annotators=annotators, timeout=timeout, memory=memory,
                                    properties=CHINESE_PROPERTIES)

    def get_feature(self, doc):
        annotated = self.client.annotate(doc)
        pos_feature = []
        ner_feature = []
        lemma_feature = []
        tokens = []
        for sentence in annotated.sentence:
            for token in sentence.token:
                tokens.append(token.word)
                pos_feature.append(token.pos)
                ner_feature.append(token.ner)
                lemma_feature.append(token.lemma)
        return {'lemma_feature': lemma_feature, 'pos': pos_feature, 'context_lower': [t.lower() for t in tokens],
                'ner': ner_feature}
