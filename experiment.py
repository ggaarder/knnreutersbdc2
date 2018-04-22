import argparse
import logging
import sys
import reuters_util as util

"""
Text Classification
Train/Test: Neuters 21578 LEWISSPLIT
Method: KNN, bdc
todo:
- remove small topics (only the eight on in the paper)
- the accuracy should at ~ 94%
- SVM
"""

def test_classify(testcorpus, traincorpus, wfunc):
    """return the predicted results list"""
    correct_cnt = 0

    for i, d in enumerate(testcorpus.DOCVECS):
        result = traincorpus.predict_with_knn(5, d, wfunc)

        if result == d.label:
            correct_cnt += 1

        logging.info('#{}: {}'.format(i, correct_cnt/(i+1)))

#        if i % 100:
#            traincorpus.write_cache_file()

        yield result

if __name__ == '__main__':
    """
    exam with TRAINCSV and TESTCSV, splited according to the LEWISSPLIT
    attribution in <NEUTERS> of the original .sgm file
    """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weight-func", type=str,
        default='tfbdc',
        choices=['tfbdc', 'tfidf'])
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    TRAINCSV = 'dat/train.csv'
    TESTCSV = 'dat/test.csv'

    traincorpus = util.csv2corpus(TRAINCSV)
    testcorpus = util.csv2corpus(TESTCSV)

    results = test_classify(testcorpus, traincorpus, args.weight_func)

    if args.output:
        out_fp = open(args.output, 'w')
    else:
        out_fp = sys.stdout

    for result in results:
        out_fp.write(result+'\n')
        out_fp.flush()

    if out_fp != sys.stdout:
        out_fp.close()