# coding:utf-8
import os

import tensorflow as tf
import collections
import six
import sys

from sogou_mrc.libraries import modeling
from sogou_mrc.libraries import tokenization
import logging


class BertDataHelper(object):
    def __init__(self, BERT_PRETRAINED_DIR='',
                 BERT_MODEL='uncased_L-12_H-768_A-12', doc_stride=128):
        DO_LOWER_CASE = BERT_MODEL.startswith('uncased')
        VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
        self.do_lower_case = True
        self.doc_stride = doc_stride
        self.tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
        self.index = 0

    def convert(self, instances,data='coqa',max_seq_length=512):
        new_data = []
        for i, instance in enumerate(instances):
            new_instance = self.convert_coqa_to_bert_input(instance, max_seq_length=max_seq_length) if data=='coqa' \
                else self.convert_to_bert_input(instance)
            if new_instance is not None and len(new_instance)>0: new_data.extend(new_instance)
        return new_data

    def convert_to_bert_input(self, instance, max_seq_length=512, max_query_length=64, is_training=True):
        tokenizer = self.tokenizer
        doc_tokens = instance['context_tokens'] if not self.do_lower_case else [token.lower() for token in
                                                                                instance['context_tokens']]

        question = instance['question'].lower() if self.do_lower_case else instance['question']
        query_tokens = tokenizer.tokenize(question)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        tok_start_position = None
        tok_end_position = None
        if is_training and  ('is_impossible' not in instance or instance['is_impossible']==0):
            tok_start_position = orig_to_tok_index[instance['answer_start']]
            if instance['answer_end'] < len(doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[instance['answer_end'] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                instance['context'].lower())
        if is_training and 'is_impossible' in instance and instance['is_impossible']==1:
            tok_start_position= -1
            tok_end_position =-1

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, self.doc_stride)
        new_instances = []
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and ('is_impossible' not in instance or instance['is_impossible']==0):
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    start_position = -1
                    end_position = -1
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and start_position == -1 and end_position == -1:
                continue
            if is_training and 'is_impossible' in instance and instance['is_impossible']==1:
                start_position = 0
                end_position = 0
            self.index += 1
            if self.index < 2:
                tf.logging.info("*** Example ***")
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))

                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                tf.logging.info('start_position: %d : end_position: %d' % (start_position, end_position))
                tf.logging.info(tokens[start_position:end_position])

            new_instance = {
                'doc_span_index': doc_span_index,
                'tokens': tokens,
                'token_to_orig_map': token_to_orig_map,
                'token_is_max_context': token_is_max_context,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'start_position': start_position,
                'end_position': end_position,
            }

            for k, v in instance.items():
                if k not in new_instance:
                    new_instance[k] = v
            new_instances.append(new_instance)
        return new_instances
    
    def convert_coqa_to_bert_input(self, instance, max_seq_length=512, max_query_length=64, is_training=True):
        tokenizer = self.tokenizer
        doc_tokens = instance['context_tokens'] if not self.do_lower_case else [token.lower() for token in
                                                                                instance['context_tokens']]

        question = instance['question'].lower() if self.do_lower_case else instance['question']
        query_tokens = tokenizer.tokenize(question)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[-max_query_length:]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        tok_start_position = None
        tok_end_position = None
        tok_rationale_start_position = None
        tok_rationale_end_position = None
        if is_training and instance['answer_type'] != "unknown":
            if instance['answer_type'] == 'extractive':
                tok_start_position = orig_to_tok_index[instance['answer_start']]
                if instance['answer_end'] < len(doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[instance['answer_end'] + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                instance['answer'])
            tok_rationale_start_position = orig_to_tok_index[instance['rationale_start']]
            if instance['rationale_end'] < len(doc_tokens) - 1:
                tok_rationale_end_position = orig_to_tok_index[instance['rationale_end'] + 1] - 1
            else:
                tok_rationale_end_position = len(all_doc_tokens) - 1
            (tok_rationale_start_position, tok_rationale_end_position) = _improve_answer_span(all_doc_tokens, tok_rationale_start_position, tok_rationale_end_position, tokenizer,instance['rationale'])

        #if is_training and instance['answer_type'] == 'unknown':
            #tok_start_position= -1
            #tok_end_position =-1

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, self.doc_stride)
        
        if is_training and len(doc_spans) > 1:
            if instance['answer_type'] in ['yes', 'no']:
                #check is there a full rationale in one chunk for yes/no answer question, if it doesn't exist, throw the example 
                full_rationale_in_one_chunk = False
                for (doc_span_index, doc_span) in enumerate(doc_spans):
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    if tok_rationale_start_position >= doc_start and tok_rationale_end_position <= doc_end:
                        full_rationale_in_one_chunk = True
                        break
                if not full_rationale_in_one_chunk:
                    return None
            if instance['answer_type'] == 'extractive':
                #check is there a full extractive answer into one chunk, if it doesn't exist, throw the example 
                full_answer_in_one_chunk = False
                for (doc_span_index, doc_span) in enumerate(doc_spans):
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    if tok_start_position >= doc_start and tok_end_position <= doc_end:
                        full_answer_in_one_chunk = True
                        break
                if not full_answer_in_one_chunk:
                    return None

        new_instances = []
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            if is_training and len(doc_spans) > 1:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if instance['answer_type'] == 'extractive': # throw chunk dosn't has full answer
                    if (tok_start_position >= doc_start and tok_start_position <= doc_end and tok_end_position > doc_end) or (tok_end_position >= doc_start and tok_end_position <= doc_end and tok_start_position < doc_start):
                        continue
                if instance['answer_type'] in ['yes', 'no']: # throw chunk dosn't has full answer
                    if (tok_rationale_start_position >= doc_start and tok_rationale_start_position <= doc_end and tok_rationale_end_position > doc_end) or (tok_rationale_end_position >= doc_start and tok_rationale_end_position <= doc_end and tok_rationale_start_position < doc_start):
                        continue
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            question_mask = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            question_mask.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
                question_mask.append(1)
            tokens.append("[SEP]")
            segment_ids.append(0)
            question_mask.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
                question_mask.append(0)
            tokens.append("[SEP]")
            segment_ids.append(1)
            question_mask.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                question_mask.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            rationale_mask = None
            no_rationale = None
            if is_training and instance['answer_type'] != 'unknown':
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                doc_offset = len(query_tokens) + 2
        
                rationale_mask = [0] * max_seq_length
                doc_rationale_start = -1
                doc_rationale_end = -1
                if tok_rationale_start_position >= doc_start and tok_rationale_start_position <= doc_end:
                    doc_rationale_start = tok_rationale_start_position - doc_start + doc_offset
                    orig_rationale_start = tok_to_orig_index[tok_rationale_start_position]
                    if tok_rationale_end_position <= doc_end:
                        doc_rationale_end = tok_rationale_end_position - doc_start + doc_offset
                    else:
                        doc_rationale_end = doc_end - doc_start + doc_offset
                else:
                    if tok_rationale_end_position >= doc_start and tok_rationale_end_position <= doc_end:
                        doc_rationale_start = doc_offset
                        doc_rationale_end = tok_rationale_end_position - doc_start + doc_offset
                if instance['answer_type'] == 'extractive':
                    out_of_span = False
                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0 
                        end_position = 0 
                        doc_rationale_start = -1
                        doc_rationale_end = -1
                        #return None #doc_rationale_end = -1
                    else:
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                        if doc_rationale_start == -1 or doc_rationale_end == -1:
                            raise ValueError(instance['qid'])
                else: # yes no
                    start_position = 0
                    end_position = 0
                if doc_rationale_start != -1 and doc_rationale_end != -1:
                    no_rationale = False 
                    ix = doc_rationale_start
                    while ix <= doc_rationale_end:
                        rationale_mask[ix] = 1
                        ix += 1
                else:
                    no_rationale = True

            if is_training and instance['answer_type']=='unknown':
                start_position = 0
                end_position = 0
                rationale_mask = [0] * max_seq_length
                no_rationale = True

            unk_mask, yes_mask, no_mask, extractive_mask = None, None, None, None
            if is_training:
                if start_position == 0 and end_position == 0 and no_rationale == True:
                    unk_mask, yes_mask, no_mask, extractive_mask = 1, 0, 0, 0
                elif start_position == 0 and end_position == 0 and no_rationale == False:
                    unk_mask, yes_mask, no_mask, extractive_mask = 0, instance['abstractive_answer_mask'][1], instance['abstractive_answer_mask'][2], 0
                elif start_position != 0 and end_position != 0 and no_rationale == False:
                    unk_mask, yes_mask, no_mask, extractive_mask = 0, 0, 0, 1

            self.index += 1
            if self.index < 2:
                logging.info("*** Example ***")
                logging.info("doc_span_index: %s" % (doc_span_index))
                logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))

                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logging.info(
                    "question_mask: %s" % " ".join([str(x) for x in question_mask]))
                logging.info('start_position: %d : end_position: %d' % (start_position, end_position))
                if is_training and instance['answer_type'] != 'unknown':
                    logging.info("answer_type: %d %d %d %d" % (unk_mask, yes_mask, no_mask, extractive_mask))
                    if extractive_mask == 1:
                        answer_text = " ".join(tokens[start_position:(end_position + 1)])
                        logging.info("start_position: %d" % (start_position))
                        logging.info("end_position: %d" % (end_position))
                        logging.info(
                          "answer: %s" % (tokenization.printable_text(answer_text)))
                    if yes_mask == 1 or no_mask == 1 or extractive_mask == 1:
                        rationale_text = " ".join(tokens[doc_rationale_start:(doc_rationale_end + 1)])
                        logging.info("rationale_start_position: %d" % (doc_rationale_start))
                        logging.info("rationale_end_position: %d" % (doc_rationale_end))
                        logging.info(
                          "rationale: %s" % (tokenization.printable_text(rationale_text)))
                        logging.info(
                            "rationale_mask: %s" % " ".join([str(x) for x in rationale_mask]))

            new_instance = {
                'doc_span_index': doc_span_index,
                'tokens': tokens,
                'token_to_orig_map': token_to_orig_map,
                'token_is_max_context': token_is_max_context,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'start_position': start_position,
                'end_position': end_position,
                'unk_mask' : unk_mask,
                'yes_mask' : yes_mask,
                'no_mask' : no_mask,
                'extractive_mask' : extractive_mask,
                'question_mask' : question_mask,
                'rationale_mask' : rationale_mask
            }

            for k, v in instance.items():
                if k not in new_instance:
                    new_instance[k] = v
            new_instances.append(new_instance)
        return new_instances

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)
