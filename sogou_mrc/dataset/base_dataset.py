# coding: utf-8


class BaseReader(object):
    def read(self, *input):
        raise NotImplementedError


class BaseEvaluator(object):
    def get_score(self, *input):
        raise NotImplementedError
