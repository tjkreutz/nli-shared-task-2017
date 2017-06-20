#!usr/bin/python3

import numpy as np
import json
import enchant
from nltk.tag import pos_tag_sents
from nltk.util import skipgrams
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


class AverageWordLength(BaseEstimator, TransformerMixin):
    """outputs average word length per document"""

    def average_word_length(self, x):
        return np.mean([len(word) for word in x.split()])

    def transform(self, X, y=None):
        return [[self.average_word_length(x)] for x in X]

    def fit(self, X, y=None):
        return self

class AverageMisspellings(BaseEstimator, TransformerMixin):
    """outputs average word length per document"""

    def average_misspellings(self, x):
        d = enchant.Dict("en_US")
        counter = 0
        toks = x.split()
        for tok in toks:
            if d.check(tok) == False:
                counter+=1

        return counter/len(toks)

    def transform(self, X, y=None):
        return [[self.average_misspellings(x)] for x in X]

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

class POSVectorizer2(TfidfVectorizer):
    """ adds postags, learns weights """

    def postag(self, X):
        closed = ["IN", "DT", "PRP", ".", ",", "CC", "TO", "PRP$"]

        tagged = pos_tag_sents([x.split() for x in X])
        new_X = []
        for doc in tagged:
            out_string = []
            for tt in doc:
                if tt[1] in closed:
                    out_string.append(tt[0])
                else:
                    out_string.append(tt[1])
            new_X.append(' '.join(out_string))

        print(new_X)
        return new_X

    def transform(self, X, y=None):
        X = self.postag(X)
        return super(POSVectorizer2, self).transform(X, y)

    def fit(self, X, y=None):
        X = self.postag(X)
        return super(POSVectorizer2, self).fit(X, y)

class LemmaVectorizer(TfidfVectorizer):
    """ adds postags, learns weights """

    def get_lemmas(self, X):

        lem = WordNetLemmatizer()
        new_X = [[lem.lemmatize(word.lower()) for word in x.split()] for x in X]
        #print(new_X)
        new_X = [' '.join(doc) for doc in new_X]
        return new_X

    def transform(self, X, y=None):
        X = self.get_lemmas(X)
        return super(LemmaVectorizer, self).transform(X, y)

    def fit(self, X, y=None):
        X = self.get_lemmas(X)
        return super(LemmaVectorizer, self).fit(X, y)


class PromptWordVectorizer(TfidfVectorizer):
    """ removes learned promptwords before training weights """

    def filterpw(self, X):
        new_X = []
        with open("keywords.json", "r") as f:
            keywords = json.load(f)
        for doc in X:
            words = doc.split()
            for t in range(len(words)):
                if words[t] in keywords:
                    words[t] = ''
            new_X.append(' '.join(words))
        return new_X

    def transform(self, X, y=None):
        X = self.filterpw(X)
        return super(PromptWordVectorizer, self).transform(X, y)

    def fit(self, X, y=None):
        X = self.filterpw(X)
        return super(PromptWordVectorizer, self).fit(X, y)


class SkipgramVectorizer(TfidfVectorizer):
    """ Learns weights for skipgrams """

    def __init__(self, n=2, k=2, base_analyzer='word', *args, **kwargs):
        super(SkipgramVectorizer, self).__init__(*args, **kwargs)
        # we only use the parent for learning weights, chars can be used but the implementation lies in our custom
        # function, not with the tfidfvectorizer
        self.n = n
        self.k = k
        self.base_analyzer = base_analyzer

    def generate_skipgrams(self, x):
        if self.base_analyzer == 'char':
            return skipgrams(x, n=self.n, k=self.k)
        else:
            return skipgrams(x.split(), n=self.n, k=self.k)

    def build_analyzer(self):
        return self.generate_skipgrams

class IPAVectorizer(TfidfVectorizer):
    """ adds postags, learns weights """

    def get_ipa(self, X):
        #new_X = [check_output(["espeak", "-q", "--ipa", '-v', 'en-us', x]).decode('utf-8') for x in X]
        new_X = []
        for x in X:
            ipa = check_output(["espeak", "-q", "--ipa=1", '-v', 'en-us', x]).decode('utf-8')
            print(X.index(x))
            new_X.append(ipa.strip())

        return new_X

    def transform(self, X, y=None):
        X = self.get_ipa(X)
        return super(IPAVectorizer, self).transform(X, y)

    def fit(self, X, y=None):
        X = self.get_ipa(X)
        return super(IPAVectorizer, self).fit(X, y)

class MisspellingVectorizer(TfidfVectorizer):
    """ Gertjan's suggested feature """

    def get_misspellings(self, X):
        d = enchant.Dict("en_US")
        new_X = [[word for word in x.split() if (not d.check(word)) and (word.isalpha())] for x in X]
        print(new_X)
        new_X = [' '.join(doc) for doc in new_X]
        return new_X

    def transform(self, X, y=None):
        X = self.get_misspellings(X)
        return super(MisspellingVectorizer, self).transform(X, y)

    def fit(self, X, y=None):
        X = self.get_misspellings(X)
        return super(MisspellingVectorizer, self).fit(X, y)

class FinalLetter(CountVectorizer):
    """ extract final letter """

    def get_letter(self, x):
        new_X = ' '.join([word[-1:] for word in x.split()])
        return new_X

    def transform(self, X, y=None):
        X = [self.get_letter(x) for x in X]
        return super(FinalLetter, self).transform(X)

    def fit(self, X, y=None):
        X = [self.get_letter(x) for x in X]
        return super(FinalLetter, self).fit(X, y)