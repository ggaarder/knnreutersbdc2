import algo

CACHE_FILE = 'dat/neuters_terms_cache.csv'
TEST_CSV = 'dat/test-8-categories.csv'
TRAIN_CSV = 'dat/train-8-categories.csv'
EXPERIMENT_FILE = 'experiment.csv'

def all_terms(train):
    all_terms = set()
    for i in train['doc']:
        all_terms |= frozenset(algo.DocVector(i).terms)

    return sorted(all_terms)