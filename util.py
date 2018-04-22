"""utilities"""
# pylint: disable=C0304
import algo

CACHE_FILE = 'dat/neuters_terms_cache.csv'
TEST_CSV = 'dat/test-8-categories.csv'
TRAIN_CSV = 'dat/train-8-categories.csv'
EXPERIMENT_FILE = 'experiment.csv'

def all_terms(train):
    """return all terms of a train data"""
    terms = set()
    for i in train['doc']:
        terms |= frozenset(algo.DocVector(i).terms)

    return sorted(terms)