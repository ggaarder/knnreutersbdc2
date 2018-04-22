import logging
import math
import multiprocessing as mp
import numpy as np
import pandas as pd

def xlogx(x):
    if x == 0:
        return 0
    return x*math.log2(x)

def categories(dat):
    return sorted(set(dat.index))

def calc_bdc_parallel(state):
    [terms_to_process, f_t_ci, f_ci, cs] = state

    # Note:
    # at the first time I initializered `out` as a pd.DataFrame,
    # and then add result by `out.loc[t, 'bdc'] = bdc`, which is
    # extremely slow since I called `loc` and tried to search by
    # `[t, 'bdc']`
    out = []

    for t in terms_to_process:
        p_t_ci = {c: f_t_ci[c][t]/f_ci[c] for c in cs}
        G_t_ci = {c: p_t_ci[c]/sum(p_t_ci.values()) for c in cs}
        F_t_ci = {c: xlogx(G_t_ci[c]) for c in cs}
        BH_t = -sum(F_t_ci.values())
        bdc = 1-BH_t/math.log2(len(cs))
        out.append(bdc)

    return pd.DataFrame(out, index=terms_to_process, columns=['bdc'])

def calc_bdc(dat, PARALLEL=4):
    cs = categories(dat)
    f_t_ci = {c: dat.loc[c].sum() for c in cs}
    f_ci = {c: f_t_ci[c].sum() for c in cs}

    states = [[terms_to_process, f_t_ci, f_ci, cs]
              for terms_to_process in np.array_split(dat.columns,
                                                     PARALLEL)]
    po = mp.Pool(processes=PARALLEL)
    res = po.map(calc_bdc_parallel, states)
    po.close()

    return pd.concat(res)
