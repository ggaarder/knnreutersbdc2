import argparse
import logging
import sys
import pandas as pd
import algo
import util
"""
Text Classification
Train/Test: Neuters 21578 LEWISSPLIT
Method: KNN, bdc
todo:
- the accuracy should at ~ 94%
- SVM
"""

def similarity(doc1, doc2, cache, wfunc):
    doc1vec, doc2vec = algo.DocVector(doc1), algo.DocVector(doc2)

    terms = set(doc1vec.terms) & set(doc2vec.terms) & set(cache['term'])
    terms = sorted(terms)

    if wfunc == 'tfidf':
        w = [float(cache[cache.term == t]['idf']) for t in terms]

    doc1vec = [doc1vec.get_tf(terms[i])*w[i] for i in range(len(terms))]
    doc2vec = [doc2vec.get_tf(terms[i])*w[i] for i in range(len(terms))]

    return algo.Vector(doc1vec).cos(algo.Vector(doc2vec))

def predict(doc, train, cache, wfunc):
    return algo.knn([
        [train.loc[i,'label'],
         similarity(doc, train.loc[i,'doc'], cache, wfunc)]
        for i in range(len(train))])

def experiment(test, train, wfunc):
    cache = pd.read_csv(util.CACHE_FILE)
    actual, expect = [], []

    out = open(util.EXPERIMENT_FILE, 'w')
    out.write('actual,expect\n')

    for i in range(len(test)):
        expect.append(test.loc[i, 'label'])
        actual.append(predict(test.loc[i, 'doc'], train, cache, wfunc))

        logging.info('#{}: {}'.format(i, algo.accuracy(actual, expect)))

        out.write('{},{}\n', actual[-1], expect[-1])
        out.flush()

    out.close()

    logging.info('Experiment success')
    logging.info('Weight Function: {}'.format(wfunc))
    logging.info('Accuracy: {}'.format(algo.accuracy(actual, expect)))
    logging.info('Micro F1: {}'.format(algo.micro_f1(actual, expect)))
    logging.info('Macro F1: {}'.format(algo.macro_f1(actual, expect)))
    logging.info('Result saved to {}'.format(util.EXPERIMENT_FILE))

if __name__ == '__main__':
    """
    exam with TRAINCSV and TESTCSV, splited according to the LEWISSPLIT
    attribution in <NEUTERS> of the original .sgm file
    """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weight-func", type=str,
        default='tfidf',
        choices=['tfbdc', 'tfidf'])
    args = parser.parse_args()

    experiment(pd.read_csv(util.TEST_CSV), pd.read_csv(util.TRAIN_CSV),
        args.weight_func)
