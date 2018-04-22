"""generate cache, like idf and bdc"""
# pylint: disable=C0103,C0304,W0640
import math
import logging
import numpy as np
import pandas as pd
import algo
import util

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train = pd.read_csv(util.TRAIN_CSV)
    corpus = algo.Corpus(train)

    try:
        cache = pd.read_csv(util.CACHE_FILE)
    except FileNotFoundError:
        cache = pd.DataFrame(
            [[i, np.nan, np.nan, np.nan] for i in corpus.TERMS],
            columns=['term', 'df', 'idf', 'bdc'])

    c = corpus.CATEGORIES
    docvecs = corpus.DOCVECS
    f_ci = [sum([len(d.split(' '))
                 for d in train[train.label == c[i]]['doc']])
            for i in range(len(c))]

    for i, t in enumerate(cache['term']):
        if pd.isna(cache.loc[i, 'df']):
            cache.loc[i, 'df'] = algo.count_if(train['doc'], lambda d: t in d)
        if pd.isna(cache.loc[i, 'idf']):
            cache.loc[i, 'idf'] = math.log(len(train)/cache.loc[i, 'df'])
        if pd.isna(cache.loc[i, 'bdc']):
            f_t_ci = [sum([d.get_tf(t) for d in docvecs[i]])
                      for i in range(len(c))]
            p_t_ci = [f_t_ci[i] / f_ci[i] for i in range(len(c))]
            G_t_ci = [p_t_ci[i] / sum(p_t_ci) for i in range(len(c))]
            F_t_ci = [algo.x_logx(G_t_ci[i]) for i in range(len(c))]
            BH_t = -sum(F_t_ci)
            cache.loc[i, 'bdc'] = 1-BH_t/math.log2(len(c))
            logging.info('%s', cache.loc[i])

    with open(util.CACHE_FILE, 'w') as o:
        o.write(cache.to_csv(index=False))