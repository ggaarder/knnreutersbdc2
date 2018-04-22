import math
import operator
from operator import itemgetter
import pandas as pd

class Vector:
    def __init__(v):
        self.v = v

    def inner_product(self, v):
        return sum(map(operator.mul, self.v, v))

    def abs(self):
        return math.sqrt(sum(map(lambda x:x**2, self.v)))

    def cos(self, v):
        E = sys.float_info.epsilon
        return self.inner_product(v)/(E+self.abs())/(E+v.abs())

class DocVector:
    """VSM"""
    def __init__(self, doc, label):
        terms = sorted(doc.split(' '))
        self.terms = sorted(set(terms))
        self.tf = {t: terms.count(t) for t in self.terms} # term frequency
        self.label = label

    def get_tf(self, term):
        """term frequency"""
        return self.tf.get(term, 0)

def avg(l):
    return sum(l)/len(l)

def x_logx(x):
    if x == 0: return 0
    return x*math.log2(x)

def majority(votes):
    labels = frozenset(votes)
    counts = [votes.count(l) for l in labels]
    return max(zip(counts, labels), key=itemgetter(0))[1]

def partition(A, key=lambda x: x):
    p, r = 0, len(A)-1
    x, i = key(A[r]), p-1

    for j in range(p, r):
        if cmp(key(A[j]), x) <= 0:
            i += 1
            A[i], A[j] = A[j], A[i]

    A[i+1], A[r] = A[r], A[i+1]

    return A[p:i+1], A[i+1], A[i+2:]

def k_smallest(k, xs, key=lambda x: x):
    return sorted(xs, key=key)[:k]

#    small, x, big = partition(xs, key)
#
#    if len(small)+1 < k:
#        return small+[x]+k_smallest(k-len(small)-1, big, key)
#    elif len(small) > k:
#        return k_smallest(k, small, key)
#    elif len(small)+1 == k:
#        return small+[x]
#    elif len(small) == k:
#        return small

def knn(k, l_s):
    """l_s for [[label, similarity]]"""
    return majority([i[0] for i in
        k_smallest(k, l_s, key=lambda x:-x[1])])

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