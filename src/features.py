#!usr/bin/python3

import numpy as np
import json
from nltk.tag import pos_tag_sents
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
    """ adds postags, learns weights """

    def postag(self, X):
        new_X = pos_tag_sents([x.split() for x in X])
        new_X = [' '.join([tt[1] for tt in doc]) for doc in new_X]
        return new_X

    def transform(self, X, y=None):
        X = self.postag(X)
        return super(POSVectorizer, self).transform(X, y)

    def fit(self, X, y=None):
        X = self.postag(X)
        return super(POSVectorizer, self).fit(X, y)

class PromptWordVectorizer(TfidfVectorizer):
    """ adds postags, learns weights """

    def filterpw(self, X):
        new_X = []
        with open("keywords.json", "r") as f:
            keywords = json.load(f)
        for doc in X:
            words = doc.split()
            for t in range(len(words)):
                if words[t] in keywords:
                    words[t] = '<PW>'
            new_X.append(' '.join(words))
        return new_X

    def transform(self, X, y=None):
        X = self.filterpw(X)
        return super(PromptWordVectorizer, self).transform(X, y)

    def fit(self, X, y=None):
        X = self.filterpw(X)
        return super(PromptWordVectorizer, self).fit(X, y)


