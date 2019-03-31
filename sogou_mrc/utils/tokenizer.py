# coding:utf-8
import spacy
from stanfordnlp.server.client import CoreNLPClient
import jieba
import multiprocessing
import re
class SpacyTokenizer(object):
    def __init__(self,fine_grained=False):
        self.nlp = spacy.load('en', disable=['parser','tagger','entity'])
        self.fine_grained = fine_grained

    def word_tokenizer(self, doc):
        if not self.fine_grained:
            doc = self.nlp(doc)
            tokens = [token.text for token in doc]
            token_spans = [(token.idx, token.idx + len(token.text)) for token in doc]
            return tokens, token_spans
        sentence = doc
        tokens = []
        token_spans = []
        cur = 0
        pattern = u'-|–|—|:|’|\.|,|\[|\?|\(|\)|~|\$|/'
        for next in re.finditer(pattern, sentence):
            for token in self.nlp(sentence[cur:next.regs[0][0]]):
                if token.text.strip() != '':
                    tokens.append(token.text)
                    token_spans.append((cur + token.idx, cur + token.idx + len(token.text)))
            tokens.append(sentence[next.regs[0][0]:next.regs[0][1]])
            token_spans.append((next.regs[0][0], next.regs[0][1]))
            cur = next.regs[0][1]
        for token in self.nlp(sentence[cur:]):
            if token.text.strip() != '':
                tokens.append(token.text)
                token_spans.append((cur + token.idx, cur + token.idx + len(token.text)))
        return tokens, token_spans

    def word_tokenizer_parallel(self, docs):
        docs = [doc for doc in self.nlp.pipe(docs, batch_size=64, n_threads=multiprocessing.cpu_count())]
        tokens = [[token.text for token in doc] for doc in docs]
        token_spans = [[(token.idx, token.idx + len(token.text)) for token in doc] for doc in docs]
        return tokens, token_spans


class JieBaTokenizer(object):
    """
    only for chinese tokenize,no pos/ner feature function
    """
    def __init__(self):
        self.tokenizer = jieba

    def word_tokenizer(self, doc):
        tokens = self.tokenizer.cut(doc)
        tokens = '<split>'.join(tokens).split('<split>')
        start = 0
        token_spans = []
        for token in tokens:
            token_spans.append((start, start + len(token)))
            start += len(token)
        return tokens, token_spans


class StanfordTokenizer(object):
    def __init__(self, language='zh', annotators='ssplit tokenize', timeout=30000, memory="4G"):
        if language=='zh':
            CHINESE_PROPERTIES = {
                "tokenize.language": "zh",
                "segment.model": "edu/stanford/nlp/models/segmenter/chinese/ctb.gz",
                "segment.sighanCorporaDict": "edu/stanford/nlp/models/segmenter/chinese",
                "segment.serDictionary": "edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz",
                "segment.sighanPostProcessing": "true",
                "ssplit.boundaryTokenRegex": "[.。]|[!?！？]+",
            }
        else:
            CHINESE_PROPERTIES = {}
        self.client = CoreNLPClient(annotators=annotators, timeout=timeout, memory=memory,properties=CHINESE_PROPERTIES)

    def word_tokenizer(self, doc):
        try:
            annotated = self.client.annotate(doc)
            tokens, token_spans = [], []
            for sentence in annotated.sentence:
                for token in sentence.token:
                    tokens.append(token.word)
                    token_spans.append((token.beginChar, token.endChar))
            return tokens, token_spans
        except Exception as e:
            return None,None