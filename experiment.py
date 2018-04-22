"""
Test knn+bdc with [reuters21578][www.daviddlewis.com/resources/testcollections/
reuters21578/reuters21578.tar.gz]
"""
import logging
import json
import math
import random
import progressbar
import numpy as np
import pandas as pd
import parse
from bdc import calc_bdc
from knn import knn

def cos(v1, v2):
    return np.dot(v1, v2)/math.sqrt(np.sum(v1**2))/math.sqrt(np.sum(v2**2))

def get_terms(ids):
    all_terms = set()
    for i in news_json:
        if i['newid'] in ids:
            all_terms |= frozenset(i['tf'].keys())
    return all_terms

def test_general(testids, trainids):
    """
    We can't speed up since:
    (1) We can't use multiprocessing to speed up, since calc_bdc used it and
    Python doesn't allow a `daemon` (i.e. a forked processes) to `have
    children`. (i.e. fork)
    (2) We don't use threads since it doesn't speed up in our case, since
    the task is CPU-bound.
    """
    logging.info('test: %d, train: %d', len(testids), len(trainids))

    trainterms = get_terms(trainids)

    logging.info('experiment begins')
    correctcnt = 0
    bar = progressbar.ProgressBar(
        widgets=[progressbar.Percentage(),
                 progressbar.Bar('>'),
                 progressbar.DynamicMessage('accuracy')],
        max_value=len(testids)).start()
    
    for no, testid in enumerate(testids):
        for n in news_json:
            if n['newid'] == testid:
                testjson = n
                break

        usedterms = sorted(get_terms([testid]) & trainterms)
        trainvec = []
        trainlabels = []
        
        for i in news_json:
            if i['newid'] in trainids:
                vec = [i['tf'].get(t, 0) for t in usedterms]
                if np.sum(vec) != 0:
                    trainvec.append(vec)
                    trainlabels.append(i['topic'])
                
        bdcs = calc_bdc(pd.DataFrame(
            trainvec,
            index=trainlabels,
            columns=usedterms))
        trainvec *= bdcs.T.values
        testvec = bdcs.T.values*[testjson['tf'].get(t, 0) for t in usedterms]
        sim = [cos(testvec, v) for v in trainvec]
        knndat = [[trainlabels[i], sim[i]] for i in range(len(sim))]
        res = knn(5, knndat)
        
        if res == testjson['topic']:
            correctcnt += 1
        bar.update(no+1, accuracy=correctcnt/(no+1))

    return correctcnt/len(testids)
        
def test_lewis():
    logging.info('figuring out lewissplit')
    testids = []
    trainids = []

    for i in news_json:
        if not i['reject']:
            if i['lewissplit'] == 'TRAIN':
                trainids.append(i['newid'])
            elif i['lewissplit'] == 'TEST':
                testids.append(i['newid'])

    return test_general(testids, trainids)

def test_random():
    """Results: 0.846 0.847"""
    VALID_NEWS_TERMS_LOWER_BOUND = 30 # skip news with too few words
    TEST_PERCENT = 20 # how big the test sample is
    testids = []
    trainids = []

    for i in news_json:
        if i['reject']:
            continue
        if len(i['tf'].keys()) < VALID_NEWS_TERMS_LOWER_BOUND:
            continue

        if random.randrange(0, 100) < TEST_PERCENT:
            testids.append(i['newid'])
        else:
            trainids.append(i['newid'])

    return test_general(testids, trainids)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    logging.info('reading news from %s', parse.NEWS_JSON)
    with open(parse.NEWS_JSON) as j:
        news_json = json.load(j)

#    logging.info('testing with lewissplit')
#    test_lewis()

    logging.info('testing with random split')
    test_random()
