import math
import logging
import pandas as pd
import numpy as np
import reuters_util as util
import knn

class Corpus:
    """immutable set of Doc Vectors"""
    CACHE_FILE = 'dat/neuters_terms_cache.csv'

    def __init__(self, docvecs):
        """preprocess for calculating bdc"""
        self.DOCVECS = docvecs
        self.CATEGORIES = None
        self.ALL_TERMS = None
        self.f_ci_cache = None
        self.select_by_category_cache = None
        self.f_t_ci_cache = None
        self.cache = { col: c.values()
            for (col, c) in
                pd.read_csv(self.CACHE_FILE, index_col=0).to_dict().items()}

    def calc_cache(self, col, index, calc):
        self.cache[col][self.terms().index(index)] = calc()

    def get_cache(self, col, index, calc):
        """template for cached calculations"""
        try:
            x = self.cache[col][index]
        except KeyError:
            # for pandas.DataFrame
            self.cache.loc[index] = [np.nan]*len(self.cache.columns)

        if pd.isna(self.cache[col][index]):
            self.calc_cache(col, index, calc)
        return self.cache[col][index]

    def terms(self):
        """{t_i}"""
        if not self.ALL_TERMS:
            self.ALL_TERMS = set()

            for d in self.DOCVECS:
                for t in d.terms:
                    self.ALL_TERMS.add(t)

            self.ALL_TERMS = sorted(list(self.ALL_TERMS))

        return self.ALL_TERMS

    def categories(self):
        """{c_i}"""
        if not self.CATEGORIES:
            self.CATEGORIES = list(frozenset(sorted(set(
                [docvec.label for docvec in self.DOCVECS]))))

        return self.CATEGORIES

    def f_ci(self, c):
        """f(c_i) = frequency sum of all terms in category c_i"""
        if not self.f_ci_cache:
            self.f_ci_cache = [None] * len(self.categories())

        i = self.categories().index(c)

        if not self.f_ci_cache[i]:
            self.f_ci_cache[i] = sum([sum(d.tf)
                for d in self.select_by_category(c)])

        return self.f_ci_cache[self.categories().index(c)]

    def select_by_category(self, category):
        if not self.select_by_category_cache:
            self.select_by_category_cache = [
                [d for d in self.DOCVECS if c == d.label]
                for c in self.categories()
            ]

        return self.select_by_category_cache[self.categories().index(category)]

    def f_t_ci(self, term, category):
        """
        f(t, c_i) = frequency of term t in category c_i

        cache because that according to profile:
           478276    0.926    0.000  101.256    0.000 predict.py:98(f_t_ci)
        cache structure is f_t_ci = f_t_ci_cache[i][t]
        """
        if not self.f_t_ci_cache:
            self.f_t_ci_cache = [None] * len(self.categories())

        i = self.categories().index(category)
        if not self.f_t_ci_cache[i]:
            self.f_t_ci_cache[i] = {}

        if term not in self.f_t_ci_cache[i]:
            r = sum([d.get_tf(term)
                for d in self.select_by_category(category)])
            self.f_t_ci_cache[i][term] = r

        return self.f_t_ci_cache[i][term]

    def p_t_ci(self, term, category):
        """p(t|c_i) = f(t, c_i) / f(c_i)"""
        return self.f_t_ci(term, category) / self.f_ci(category)

    def sum_p_t_ci(self, term):
        """\sum_{i=1}{|C|} p(t|c_i)"""
        return sum([self.p_t_ci(term, c) for c in self.categories()])

    def G_t_ci(self, term, category):
        """G(t, c_i) = p(t|c_i)/(\sum_{i=1}{|C|} p(t|c_i))"""
        return self.p_t_ci(term, category) / self.sum_p_t_ci(term)

    def F_t_ci(self, term, category):
        """F(t, c_i) = G(t, c_i) log G(t, c_i)"""
        return util.x_logx(self.G_t_ci(term, category))

    def BH_t(self, term):
        """BH(t) = -\sum{i=1}{|C|} F(t, c_i)"""
        return -sum([self.F_t_ci(term, c) for c in self.categories()])

    def bdc(self, term):
        """bdc(t) = 1 - BH(t)/log(|C|)"""
        return self.get_cache('bdc', term,
            lambda: 1-self.BH_t(term)/math.log2(len(self.categories())))

    def df(self, term):
        """a term's document frequency"""
        return self.get_cache('df', term,
            lambda: len([d for d in self.DOCVECS if term in d.terms]))

    def idf(self, term):
        """
        [Tf-idf __ A Single-Page Tutorial - Information Retrieval and Text
        Mining][www.tfidf.com]
        -----------------------------------------------------------------------
        IDF: Inverse Document Frequency, which measures how important a term
        is. While computing TF, all terms are considered equally important.
        However it is known that certain terms, such as "is", "of", and
        "that", may appear a lot of times but have little importance. Thus we
        need to weigh down the frequent terms while scale up the rare ones, by
        computing the following:

        IDF(t) = log_e(Total number of documents / Number of documents with
        term t in it).
        -----------------------------------------------------------------------
        """
        return self.get_cache('idf', term,
            lambda: math.log(len(self.DOCVECS)/self.df(term)))

    def tfbdc(self, t, doc):
        """get the tfbdc of t respected to doc"""
        return doc.get_tf(t)*self.bdc(t)

    def tfidf(self, t, doc):
        return doc.get_tf(t)*self.idf(t)

    def weight(self, term, doc, wfunc):
        """wrapper"""
        return (getattr(self, wfunc))(term, doc)

    def predict_with_knn(self, knn_k_value, d, wfunc):
        """
        return d's predicted label
        wfunc = weight function = tfbdc, ...
        """
        logging.info('Weighting docvec')

        # remove terms that in the test-corpus however not in train-corpus
        whitelst = [t for t in d.terms if t in self.terms()]

        dw = [self.weight(t, d, wfunc) for t in whitelst] # weighted copy

        """
        weighted vectors of train data
        0. strip unused terms. only respect those terms occurs in
            docvec_to_predict
        1. weight with bdc
        """
        twv = [[self.weight(t, doc, wfunc) for t in whitelst]
            for doc in self.DOCVECS]

        labels = [d.label for d in self.DOCVECS]

        logging.info('Using KNN to classify')
        return knn.knn_classify(5, dw, twv, labels)