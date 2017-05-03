#!usr/bin/python3

import numpy as np
from nltk.tag import tnt
from nltk.corpus import treebank
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class AverageWordLength(BaseEstimator, TransformerMixin):
    """outputs average word length per document"""

    def average_word_length(self, x):
        return np.mean([len(word) for word in x.split()])

    def transform(self, X, y=None):
        return [[self.average_word_length(x)] for x in X]

    def fit(self, X, y=None):
        return self


class POSVectorizer(TfidfVectorizer):
    """outputs average word length per document"""

    def __init__(self, *args, **kwargs):
        super(POSVectorizer, self).__init__(*args, **kwargs)
        self.postagger = self.train_postagger()

    def train_postagger(self):
        t = tnt.TnT()
        t.train(treebank.tagged_sents())
        return t

    def postag(self, x):
        return [tt[1] for tt in self.postagger.tag(x)]

    def transform(self, X, y=None):
        X = [[self.postag(x)] for x in X]
        return super(POSVectorizer, self).transform(X, y)

    def fit(self, X, y=None):
        return self