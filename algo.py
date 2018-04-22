"""miscellaneous algorithms"""
import math
import operator
from operator import itemgetter
import sys
import pandas as pd

class Vector:
    """vector"""
    def __init__(self, v):
        self.v = v

    def inner_product(self, v):
        """dot product"""
        return sum(map(operator.mul, self.v, v.v))

    def abs(self):
        """abs"""
        return math.sqrt(sum(map(lambda x:x**2, self.v)))

    def cos(self, v):
        """cosine"""
        E = sys.float_info.epsilon
        return self.inner_product(v)/(E+self.abs())/(E+v.abs())

class DocVector:
    """VSM"""
    def __init__(self, doc, label=None):
        terms = sorted(doc.split(' '))
        self.terms = sorted(set(terms))
        self.tf = {t: terms.count(t) for t in self.terms} # term frequency
        self.label = label

    def get_tf(self, term):
        """term frequency"""
        return self.tf.get(term, 0)

class Corpus:
    """set of DocVector"""
    def __init__(self, dat):
        """dat is pandas DataFrame"""
        self.DAT = dat
        self.TERMS = self.all_terms()
        self.CATEGORIES = sorted(set(self.DAT['label']))
        self.DOCVECS = self.docvecs()

    def docvecs(self):
        """
        split by categories
        a great optimization (test v6, which calculates bdc ~ 2sec/word because
        it calls DocVector everytime it encounters a term)
        """
        c = self.CATEGORIES
        return [[DocVector(d) for d in self.DAT[self.DAT.label == c[i]]['doc']]
                for i in range(len(c))]

    def all_terms(self):
        """return all terms of a train data"""
        terms = set()
        for i in self.DAT['doc']:
            terms |= frozenset(DocVector(i).terms)

        return sorted(terms)

def avg(l):
    """average"""
    return sum(l)/len(l)

def x_logx(x):
    """0log0 = 0"""
    if x == 0: return 0
    return x*math.log2(x)

def majority(votes):
    """vote"""
    labels = frozenset(votes)
    counts = [votes.count(l) for l in labels]
    return max(zip(counts, labels), key=itemgetter(0))[1]

def partition(A, key=lambda x: x):
    """Quicksort Partition"""
    p, r = 0, len(A)-1
    x, i = key(A[r]), p-1

    for j in range(p, r):
        if key(A[j]) <= x:
            i += 1
            A[i], A[j] = A[j], A[i]

    A[i+1], A[r] = A[r], A[i+1]

    return A[p:i+1], A[i+1], A[i+2:]

def naive_top_k(k, s, key=lambda x: x):
    return sorted(s, key=key)[:k]

def top_k(k, xs, key=lambda x: x):
    small, pivot, big = partition(xs, key)

    if len(small) == k:
        return small
    if len(small)+1 == k:
        return small+[pivot]

    # len(small) != k and len(small) != k-1

    if len(small) > k:
        return top_k(k, small, key)

    # len(small) < k and len(small) != k-1
    # i.e. len(small) < k-1 i.e. len(small)+1 < k
    return small+[pivot]+top_k(k-len(small)-1, big, key)

def knn(k, l_s):
    """l_s for [[label, similarity]]"""
    return majority([i[0] for i in top_k(k, l_s, key=lambda x:-x[1])])

def count_if(xs, f):
    return len([None for x in xs if f(x)])

def accuracy(actual, expect):
    return count_if(zip(actual, expect), lambda x: x[0] == x[1])/len(expect)

def f1(p, r):
    return 2/(1/p+1/r)

def precision(A_cnt, B_cnt, C_cnt, D_cnt):
    return A_cnt/(A_cnt+C_cnt)

def recall(A_cnt, B_cnt, C_cnt, D_cnt):
    return A_cnt/(A_cnt+B_cnt)

def category_ABCD(c, actual, expected):
    # True Positive and classifier Positive
    tp_cp = [None for [resu, corr] in zip(actual, expected)
        if corr == c and resu == c ]
    A_cnt = len(tp_cp)

    # True Positive and Classifier Negative
    tp_cn = [None for [resu, corr] in zip(actual, expected)
        if corr == c and resu != c ]
    B_cnt = len(tp_cn)

    # True Negative and Classifier Positive
    tn_cp = [None for [resu, corr] in zip(actual, expected)
        if corr != c and resu == c ]
    C_cnt = len(tn_cp)

    # True Negative and Classifier Negative
    tn_cn = [None for [resu, corr] in zip(actual, expected)
        if corr != c and resu != c ]
    D_cnt = len(tn_cn)

    return A_cnt, B_cnt, C_cnt, D_cnt

def global_ABCD(actual, expected):
    # True Positive and classifier Positive
    tp_cp = [None for [resu, corr] in zip(actual, expected)
        if corr == resu]
    A_cnt = len(tp_cp)

    # True Positive and Classifier Negative
    tp_cn = [None for [resu, corr] in zip(actual, expected)
        if corr != resu  ]
    B_cnt = len(tp_cn)

    # True Negative and Classifier Positive
    tn_cp = [None for [resu, corr] in zip(actual, expected)
        if corr != resu ]
    C_cnt = len(tn_cp)

    # True Negative and Classifier Negative
    tn_cn = [None for [resu, corr] in zip(actual, expected)
        if corr == resu ]
    D_cnt = len(tn_cn)

    return A_cnt, B_cnt, C_cnt, D_cnt

def micro_f1(actual, expected):
    abcd = global_ABCD(actual, expected)
    return f1(precision(*abcd), recall(*abcd))

def macro_f1(actual, expected):
    dat = pd.DataFrame(
        [ [c, expected.count(c), f1(*precision_recall(*ABCD_category(c)))]
          for c in sorted(set(actual)) ],
        columns=['category', 'category_frequency', 'f1-macro'])

    macro_f1 = avg(dat['f1-macro'].dropna(how='any'))

    return macro_f1, micro_f1