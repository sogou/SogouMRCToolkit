# coding:utf-8
from sogou_mrc.dataset.base_dataset import BaseReader, BaseEvaluator
from tqdm import tqdm
import logging
import json
from sogou_mrc.utils.tokenizer import StanfordTokenizer
from collections import OrderedDict
import re


class CMRCReader(BaseReader):
    def __init__(self):
        self.tokenizer = StanfordTokenizer(language='zh')

    def read(self, file_path):
        logging.info("Reading file at %s", file_path)
        logging.info("Processing the dataset.")
        instances = self._read(file_path)
        instances = [instance for instance in tqdm(instances)]
        return instances

    def _read(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as dataset_file:
            dataset = json.load(dataset_file)
        for article in dataset:
            context = article["context_text"].replace(' ','')
            context = Space.remove_white_space(context)
            context_tokens, context_token_spans = self.tokenizer.word_tokenizer(context)
            if context_tokens is None and context_token_spans is None:continue
            for question_answer in article['qas']:
                question = question_answer['query_text']
                question_tokens, _ = self.tokenizer.word_tokenizer(question)
                answers, span_starts, span_ends = [], [], []
                if "answers" in question_answer:
                    answers = [str(answer) for answer in question_answer['answers']]
                    try:

                        span_starts = [context.index(str(answer)) for answer in question_answer['answers'] if context.index(str(answer))>0]
                        span_ends = [start + len(str(answer)) for start, answer in zip(span_starts, answers)]
                    except Exception as e:
                        continue
                answer_char_spans = zip(span_starts, span_ends) if len(
                    span_starts) > 0 and len(span_ends) > 0 else None
                if answer_char_spans is None : continue 
                answers = answers if len(answers) > 0 else None
                query_id = question_answer['query_id']

                yield self._make_instance(context, context_tokens, context_token_spans,
                                          question, question_tokens, answer_char_spans, answers, query_id)

    def _make_instance(self, context, context_tokens, context_token_spans, question, question_tokens,
                       answer_char_spans=None, answers=None, query_id=None):
        answer_token_starts, answer_token_ends = [], []
        if answers is not None:
            for answer_char_start, answer_char_end in answer_char_spans:
                answer_token_span = self._find_ans_start_end(context_tokens, answer_char_start, answer_char_end)

                assert len(answer_token_span) > 0
                answer_token_starts.append(answer_token_span[0])
                answer_token_ends.append(answer_token_span[-1])
        return OrderedDict({
            "context": context,
            "query_id": query_id,
            "context_tokens": context_tokens,
            "context_token_spans": context_token_spans,
            "question": question,
            "question_tokens": question_tokens,
            "answer": answers[0] if answers is not None else None,
            "answer_start": answer_token_starts[0] if answers is not None else None,
            "answer_end": answer_token_ends[0] if answers is not None else None,
        })

    def _find_ans_start_end(self, context_tokens, char_answer_start, char_answer_end):
        # find answer start position

        pos_s = 0
        pos_e = 0
        is_find_s = False

        # find start and end position
        tmp_len = 0
        for token_i, token in enumerate(context_tokens):
            tmp_len += len(token)
            if not is_find_s and tmp_len - 1 >= char_answer_start:
                pos_s = token_i
                is_find_s = True
            if tmp_len - 1 >= char_answer_end:
                pos_e = token_i
                break
        if pos_e == 0 and pos_s > 0: pos_e = len(context_tokens)

        return (pos_s, pos_e - 1)


def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


# remove punctuation
def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


stanford_tokenizer = StanfordTokenizer(language='zh')


# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search('[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss, _ = stanford_tokenizer.word_tokenizer(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss, _ = stanford_tokenizer.word_tokenizer(temp_str)
        segs_out.extend(ss)

    return segs_out


class CMRCEvaluator(BaseEvaluator):
    def __init__(self, file_path, monitor='f1'):
        self.ground_dict = dict()
        self.id_list = []
        self.monitor = monitor

        with open(file_path, 'r', encoding='utf-8') as dataset_file:
            dataset = json.load(dataset_file)

        for article in dataset:
            context = article["context_text"].replace(' ', '')
            context = Space.remove_white_space(context)
            context_tokens, context_token_spans = stanford_tokenizer.word_tokenizer(context)
            if context_tokens is None and context_token_spans is None: continue
            for question_answer in article['qas']:
                span_starts,span_ends=[],[]
                if "answers" in question_answer:
                    answers = [str(answer).strip() for answer in question_answer['answers']]
                    try:
                        span_starts = [context.index(str(answer)) for answer in question_answer['answers'] if context.index(str(answer))>0]
                        span_ends = [start + len(str(answer)) for start, answer in zip(span_starts, answers)]
                    except Exception as e:
                        continue
                if len(span_starts)==0 and len(span_ends)==0: continue #id = question_answer["query_id"]
                id = question_answer["query_id"]
                self.ground_dict[id] = [str(answer) for answer in question_answer['answers']]
                self.id_list.append(id)

    def get_monitor(self):
        return self.monitor

    def get_score(self, pred_answer):
        if isinstance(pred_answer, list):
            assert len(self.id_list) == len(pred_answer)
            answer_dict = dict(zip(self.id_list, pred_answer))
        else:
            answer_dict = pred_answer
        f1 = em = total = 0
        for key, value in answer_dict.items():
            ground_truths = self.ground_dict[key]
            prediction = value
            total += 1
            f1 += calc_f1_score(ground_truths, prediction)
            em += calc_em_score(ground_truths, prediction)
        exact_match = 100.0 * em / total
        f1 = 100.0 * f1 / total
        return {'exact_match': exact_match, 'f1': f1}

class Space:
    WHITE_SPACE = ' \t\n\r\u00A0\u1680​\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a' \
                  '​​\u202f\u205f​\u3000\u2028\u2029'

    @staticmethod
    def is_white_space(c):
        return c in Space.WHITE_SPACE

    @staticmethod
    def remove_white_space(s):
        return re.sub('['+Space.WHITE_SPACE+']', '', s)
