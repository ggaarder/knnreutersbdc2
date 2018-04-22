import operator
import numpy as np
import pandas as pd
import algo
import docvector
import util

def all_terms(train_csv):
    all_terms = set()
    for i in train_csv['doc']:
        all_terms |= frozenset(docvector.DocVector(i, None).terms)

    return sorted(all_terms)

def df(t, train_csv):
    return algo.count_if(train_csv['doc'], lambda d: t in d)

if __name__ == '__main__':
    train_csv = pd.read_csv(util.TRAIN_CSV)

    try:
        cache = pd.read_csv(util.CACHE_FILE)
    except FileNotFoundError:
        cache = pd.DataFrame(
            [[i, np.nan, np.nan, np.nan] for i in all_terms(train_csv)],
            columns=['term', 'df', 'idf', 'bdc'])

    # calc df
    for i,t in enumerate(cache['term']):
        if not pd.isna(cache['df'][i]):
            cache['df'][i] = df(t, train_csv)

    with open(util.CACHE_FILE, 'w') as o:
        o.write(cache.to_csv())