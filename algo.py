import math
import operator
from operator import itemgetter

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