"""
Text Classification

Train/Test: Neuters 21578 LEWISSPLIT
exam with TRAINCSV and TESTCSV, splited according to the LEWISSPLIT
attribution in <NEUTERS> of the original .sgm file

Method: KNN, bdc
todo:
- the accuracy should at ~ 94%
- SVM
"""

import argparse
import logging
import pandas as pd
import algo
import util

def similarity(doc1, doc2, cache, wfunc):
    """cosine"""
    doc1vec, doc2vec = algo.DocVector(doc1), algo.DocVector(doc2)

    terms = set(doc1vec.terms) & set(doc2vec.terms) & set(cache['term'])
    terms = sorted(terms)

    if wfunc == 'tfidf':
        weights = [float(cache[cache.term == t]['idf']) for t in terms]
    elif wfunc == 'tfbdc':
        weights = [float(cache[cache.term == t]['bdc']) for t in terms]

    doc1vec = [doc1vec.get_tf(terms[i])*weights[i] for i in range(len(terms))]
    doc2vec = [doc2vec.get_tf(terms[i])*weights[i] for i in range(len(terms))]

    return algo.Vector(doc1vec).cos(algo.Vector(doc2vec))

def predict(doc, train, cache, wfunc):
    """predict doc's category"""
    return algo.knn(5, [
        [train.loc[i, 'label'],
         similarity(doc, train.loc[i, 'doc'], cache, wfunc)]
        for i in range(len(train))])

def experiment(test, train, wfunc):
    """test"""
    cache = pd.read_csv(util.CACHE_FILE)
    actual, expect = [], []

    out = open(util.EXPERIMENT_FILE, 'w')
    out.write('actual,expect\n')

    for i in range(len(test)):
        expect.append(test.loc[i, 'label'])
        actual.append(predict(test.loc[i, 'doc'], train, cache, wfunc))

        logging.info('#%d: %f', i, algo.accuracy(actual, expect))

        out.write('{},{}\n'.format(actual[-1], expect[-1]))
        out.flush()

    out.close()

    logging.info('Experiment success')
    logging.info('Weight Function: %s', wfunc)
    logging.info('Accuracy: %f', algo.accuracy(actual, expect))
    logging.info('Micro F1: %f', algo.micro_f1(actual, expect))
    logging.info('Macro F1: %f', algo.macro_f1(actual, expect))
    logging.info('Result saved to %s', util.EXPERIMENT_FILE)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-w", "--weight-func", type=str,
                        default='tfbdc',
                        choices=['tfbdc', 'tfidf'])
    ARGS = PARSER.parse_args()

    experiment(pd.read_csv(util.TEST_CSV), pd.read_csv(util.TRAIN_CSV),
               ARGS.weight_func)
