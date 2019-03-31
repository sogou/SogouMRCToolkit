# coding:utf-8

from sogou_mrc.utils.tokenizer import SpacyTokenizer
from sogou_mrc.dataset.base_dataset import BaseReader, BaseEvaluator
import json
from collections import OrderedDict, Counter
from tqdm import tqdm
import logging
import re
import collections
import string
import sys


class SquadV2Reader(BaseReader):
    def __init__(self):
        self.tokenizer = SpacyTokenizer()

    def read(self, file_path):
        logging.info("Reading file at %s", file_path)
        logging.info("Processing the dataset.")
        instances = self._read(file_path)
        instances = [instance for instance in tqdm(instances)]
        return instances

    def _read(self, file_path):
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        for article in dataset:
            for paragraph in article['paragraphs']:
                context = paragraph["context"]
                context_tokens, context_token_spans = self.tokenizer.word_tokenizer(context)
                for question_answer in paragraph['qas']:
                    question = question_answer["question"].strip()
                    question_tokens, _ = self.tokenizer.word_tokenizer(question)

                    answers, span_starts, span_ends = [], [], []
                    if "answers" in question_answer:
                        answers = [answer['text'] for answer in question_answer['answers']]
                        span_starts = [answer['answer_start']
                                       for answer in question_answer['answers']]
                        span_ends = [start + len(answer)
                                     for start, answer in zip(span_starts, answers)]
                    if 'is_impossible' in question_answer and question_answer['is_impossible']:
                        span_starts = [0]
                        span_ends = [0]

                    answer_char_spans = zip(span_starts, span_ends) if len(
                        span_starts) > 0 and len(span_ends) > 0 else None
                    answers = answers if len(answers) > 0 else [""]
                    is_impossible = None if 'is_impossible' not in question_answer else question_answer[
                                                                                            'is_impossible'] * 1
                    qid = question_answer['id']
                    yield self._make_instance(context, context_tokens, context_token_spans,
                                              question, question_tokens, answer_char_spans, answers, is_impossible, qid)

    def _make_instance(self, context, context_tokens, context_token_spans, question, question_tokens,
                       answer_char_spans=None, answers=None, is_impossible=None, qid=None):
        answer_token_starts, answer_token_ends = [], []
        if answers is not None:
            for answer_char_start, answer_char_end in answer_char_spans:
                answer_token_span = []
                for idx, span in enumerate(context_token_spans):
                    if not (answer_char_end <= span[0] or answer_char_start >= span[1]):
                        answer_token_span.append(idx)

                if  len(answer_token_span) == 0: break #print(is_impossible) #break #continue
                answer_token_starts.append(answer_token_span[0])
                answer_token_ends.append(answer_token_span[-1])
        abstractive_answer_mask = [0]
        if is_impossible is not None and is_impossible:
            answer_token_starts=[]
            answer_token_ends=[]
            answer_token_starts.append(0)
            answer_token_ends.append(0)
            abstractive_answer_mask = [1]

        return OrderedDict({
            "context": context,
            "context_tokens": context_tokens,
            "context_token_spans": context_token_spans,
            "is_impossible": is_impossible,
            "question": question,
            'qid': qid,
            "question_tokens": question_tokens,
            "answer": answers[0] if answers is not None else [],
            "answer_start": answer_token_starts[0] if len(answer_token_starts) > 0 is not None else None,
            "answer_end": answer_token_ends[0] if len(answer_token_ends) > 0 is not None else None,
            "abstractive_answer_mask": abstractive_answer_mask
        })


class SquadV2Evaluator(BaseEvaluator):
    def __init__(self, file_path, monitor='best_f1', na_prob_thresh=1.0):
        self.na_prob_thresh = na_prob_thresh
        self.monitor = monitor
        self.ground_dict = dict()
        self.id_list = []
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        self.has_ans_qids = set()
        self.no_ans_qids = set()
        for article in dataset:
            for paragraph in article['paragraphs']:
                for question_answer in paragraph['qas']:
                    id = question_answer["id"]
                    gold_answers = [a['text'] for a in question_answer['answers']
                                    if SquadV2Evaluator.normalize_answer(a['text'])]
                    if not gold_answers:
                        gold_answers = ['']
                        self.no_ans_qids.add(id)
                    else:
                        self.has_ans_qids.add(id)
                    self.ground_dict[id] = gold_answers

    def get_monitor(self):
        return self.monitor

    def get_score(self, input):
        preds, na_proba = input
        exact_raw, f1_raw = self.get_raw_score(preds)
        na_prob_thresh = self.na_prob_thresh
        exact_thresh = self.apply_no_ans_threshold(exact_raw, na_proba, na_prob_thresh)
        f1_thresh = self.apply_no_ans_threshold(f1_raw, na_proba, na_prob_thresh)
        out_eval = SquadV2Evaluator.make_eval_dict(exact_thresh, f1_thresh)
        if len(self.has_ans_qids) > 0:
            has_ans_eval = SquadV2Evaluator.make_eval_dict(exact_thresh, f1_thresh, qid_list=self.has_ans_qids)
            SquadV2Evaluator.merge_eval(out_eval, has_ans_eval, 'HasAns')
        if len(self.no_ans_qids) > 0:
            no_ans_eval = SquadV2Evaluator.make_eval_dict(exact_thresh, f1_thresh, qid_list=self.no_ans_qids)
            SquadV2Evaluator.merge_eval(out_eval, no_ans_eval, 'NoAns')
        self.find_all_best_thresh(out_eval, preds, exact_raw, f1_raw, na_proba)
        return dict(out_eval)

    def find_all_best_thresh(self, main_eval, preds, exact_raw, f1_raw, na_probs):
        best_exact, exact_thresh = self.find_best_thresh(preds, exact_raw, na_probs)
        best_f1, f1_thresh = self.find_best_thresh(preds, f1_raw, na_probs)
        main_eval['best_exact'] = best_exact
        main_eval['best_exact_thresh'] = exact_thresh
        main_eval['best_f1'] = best_f1
        main_eval['best_f1_thresh'] = f1_thresh

    @staticmethod
    def merge_eval(main_eval, new_eval, prefix):
        for k in new_eval:
            main_eval['%s_%s' % (prefix, k)] = new_eval[k]

    @staticmethod
    def make_eval_dict(exact_scores, f1_scores, qid_list=None):
        if not qid_list:
            total = len(exact_scores)
            return collections.OrderedDict([
                ('exact', 100.0 * sum(exact_scores.values()) / total),
                ('f1', 100.0 * sum(f1_scores.values()) / total),
                ('total', total),
            ])
        else:
            total = len(qid_list)
            return collections.OrderedDict([
                ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ('total', total),
            ])

    def find_best_thresh(self, preds, scores, na_probs):
        num_no_ans = len(self.no_ans_qids)
        cur_score = num_no_ans
        best_score = cur_score
        best_thresh = 0.0
        qid_list = sorted(na_probs, key=lambda k: na_probs[k])
        for i, qid in enumerate(qid_list):
            if qid not in scores: continue
            if qid in self.has_ans_qids:
                diff = scores[qid]
            else:
                if preds[qid]:
                    diff = -1
                else:
                    diff = 0
            cur_score += diff
            if cur_score > best_score:
                best_score = cur_score
                best_thresh = na_probs[qid]
        return 100.0 * best_score / len(scores), best_thresh

    def apply_no_ans_threshold(self, scores, na_probs, na_prob_thresh):
        new_scores = {}
        for qid, s in scores.items():
            pred_na = na_probs[qid] > na_prob_thresh
            if pred_na:
                new_scores[qid] = float(not qid in self.has_ans_qids)
            else:
                new_scores[qid] = s
        return new_scores

    def get_raw_score(self, preds):
        # assert len(self.ground_dict) == len(preds)
        exact_scores = {}
        f1_scores = {}
        for qid in self.ground_dict:
            if qid not in preds:
                print('Missing prediction for %s' % qid)
                continue
            a_pred = preds[qid]
            golden_answers = self.ground_dict[qid]
            # Take max over all gold answers
            exact_scores[qid] = max([SquadV2Evaluator.compute_exact(a, a_pred) for a in golden_answers])
            f1_scores[qid] = max([SquadV2Evaluator.compute_f1(a, a_pred) for a in golden_answers])
        return exact_scores, f1_scores

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def compute_exact(a_gold, a_pred):
        return int(SquadV2Evaluator.normalize_answer(a_gold) == SquadV2Evaluator.normalize_answer(a_pred))

    @staticmethod
    def get_tokens(s):
        if not s: return []
        return SquadV2Evaluator.normalize_answer(s).split()

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = SquadV2Evaluator.get_tokens(a_gold)
        pred_toks = SquadV2Evaluator.get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
