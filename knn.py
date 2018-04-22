from operator import itemgetter

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

def top_k(k, xs, key=lambda x: x):
    """much faster than sorted(xs, key=key)[:k]"""
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

def majority(votes):
    """vote"""
    labels = frozenset(votes)
    counts = [votes.count(l) for l in labels]
    return max(zip(counts, labels), key=itemgetter(0))[1]

def knn(k, l_s):
    """l_s for [[label, similarity]]"""
    return majority([i[0] for i in top_k(k, l_s, key=lambda x:-x[1])])
