import pandas as pd
import algo

"""
see bdc [17]: Beyond tfidf weighting for text categorization in the vector
space model. Pascal Soucy and Guy W Mineau. In IJCAI, volume 5, pages
1130–1135, 2005.
--------------------------------------------------------------------------
                      Classifier      Classifier
                     positive label  negative label
True positive label       A               B
True negative label       C               D

For any category, the classifier precision is defined as A/(A+C) and the
recall as A/(A+B).

F1 = (p+r)/(2*p*r)

The micro-F1 is the F1 in (10) where A, B, C and D are global values
instead of categorybased ones. For instance, A in the micro-F1 is the
total number of classifications made by the n classifiers that were good
predictions.
---------------------------------------------------------------------------

note: according to this paper F1 = (p+r)/(2pr) however according to
[Wikipedia][https://en.wikipedia.org/wiki/F1_score]:
---------------------------------------------------------------------------
F1 = 2/(1/p+1/r) = 2pr/(p+r)
---------------------------------------------------------------------------
"""

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

    macro_f1 = algo.avg(dat['f1-macro'].dropna(how='any'))

    return macro_f1, micro_f1