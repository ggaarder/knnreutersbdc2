import math
import operator
import numpy as np
import pandas as pd
import algo
import util

if __name__ == '__main__':
    train_csv = pd.read_csv(util.TRAIN_CSV)

    try:
        cache = pd.read_csv(util.CACHE_FILE)
    except FileNotFoundError:
        cache = pd.DataFrame(
            [[i, np.nan, np.nan, np.nan] for i in util.all_terms(train_csv)],
            columns=['term', 'df', 'idf', 'bdc'])

    for i,t in enumerate(cache['term']):
        if pd.isna(cache.loc[i, 'df']):
            cache.loc[i, 'df'] = algo.count_if(train_csv['doc'],
                lambda d: t in d)
        if pd.isna(cache.loc[i, 'idf']):
            cache.loc[i, 'idf'] = math.log(len(train_csv)/cache.loc[i, 'df'])

    with open(util.CACHE_FILE, 'w') as o:
        o.write(cache.to_csv(index=False))