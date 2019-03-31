# coding: utf-8
from sogou_mrc.utils.tokenizer import SpacyTokenizer
from sogou_mrc.dataset.base_dataset import BaseReader, BaseEvaluator
import json
from collections import OrderedDict, Counter
from tqdm import tqdm
import logging
import re
import string


class SquadReader(BaseReader):
    def __init__(self,fine_grained = False):
        self.tokenizer = SpacyTokenizer(fine_grained)

    def read(self, file_path):
        logging.info("Reading file at %s", file_path)
        logging.info("Processing the dataset.")
        instances = self._read(file_path)
        instances = [instance for instance in tqdm(instances)]
        return instances

    def _read(self, file_path, context_limit=-1):
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

                    answer_char_spans = zip(span_starts, span_ends) if len(
                        span_starts) > 0 and len(span_ends) > 0 else None
                    answers = answers if len(answers) > 0 else None
                    qid = question_answer['id']
                    instance = self._make_instance(context, context_tokens, context_token_spans,
                                                   question, question_tokens, answer_char_spans, answers,qid)
                    if len(instance['context_tokens']) > context_limit and context_limit > 0:
                        if instance['answer_start'] > context_limit or instance['answer_end'] > context_limit:
                            continue
                        else:
                            instance['context_tokens'] = instance['context_tokens'][:context_limit]
                    yield instance

    def _make_instance(self, context, context_tokens, context_token_spans, question, question_tokens,
                       answer_char_spans=None, answers=None,qid=None):
        answer_token_starts, answer_token_ends = [], []
        if answers is not None:
            for answer_char_start, answer_char_end in answer_char_spans:
                answer_token_span = []
                for idx, span in enumerate(context_token_spans):
                    if not (answer_char_end <= span[0] or answer_char_start >= span[1]):
                        answer_token_span.append(idx)

                assert len(answer_token_span) > 0
                answer_token_starts.append(answer_token_span[0])
                answer_token_ends.append(answer_token_span[-1])

        return OrderedDict({
            "context": context,
            "context_tokens": context_tokens,
            "context_token_spans": context_token_spans,
            "context_word_len": [len(word) for word in context_tokens ],
            "question_word_len": [len(word) for word in question_tokens ],
            "question": question,
            'qid':qid,
            "question_tokens": question_tokens,
            "answer": answers[0] if answers is not None else None,
            "answer_start": answer_token_starts[0] if answers is not None else None,
            "answer_end": answer_token_ends[0] if answers is not None else None,
        })


class SquadEvaluator(BaseEvaluator):
    def __init__(self, file_path, monitor='f1'):
        self.ground_dict = dict()
        self.id_list = []
        self.monitor = monitor

        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        for article in dataset:
            for paragraph in article['paragraphs']:
                for question_answer in paragraph['qas']:
                    id = question_answer["id"]
                    self.ground_dict[id] = [answer['text'] for answer in question_answer['answers']]
                    self.id_list.append(id)

    def get_monitor(self):
        return self.monitor

    def get_score(self, pred_answer):
        if isinstance(pred_answer, list):
            assert len(self.id_list) == len(pred_answer)
            answer_dict = dict(zip(self.id_list, pred_answer))
        else:
            answer_dict = pred_answer

        f1 = exact_match = total = 0
        for key, value in answer_dict.items():
            total += 1
            ground_truths = self.ground_dict[key]
            prediction = value
            exact_match += SquadEvaluator.metric_max_over_ground_truths(
                SquadEvaluator.exact_match_score, prediction, ground_truths)
            f1 += SquadEvaluator.metric_max_over_ground_truths(
                SquadEvaluator.f1_score, prediction, ground_truths)
        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        return {'exact_match': exact_match, 'f1': f1}

    @staticmethod
    def normalize_answer(s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return (SquadEvaluator.normalize_answer(prediction) == SquadEvaluator.normalize_answer(ground_truth))

    @staticmethod
    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    @staticmethod
    def f1_score(prediction, ground_truth):
        prediction_tokens = SquadEvaluator.normalize_answer(prediction).split()
        ground_truth_tokens = SquadEvaluator.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
