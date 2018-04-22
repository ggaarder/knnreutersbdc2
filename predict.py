import copy
import csv
import math
import logging
import operator
import itertools

"""
Text Classification
Train/Test: Neuters 21578 LEWISSPLIT
Method: KNN, bdc
todo: SVM, tf*bdc, idf, etc.
"""

BDC_CACHE_FILE = 'bdc.cache'

def load_bdc_cache():
    # logging.info('Reading bdc cache')
    return {t: float(b) for t, b in read_csv(BDC_CACHE_FILE)}

def write_bdc_cache(cache):
    # logging.info('Writing bdc cache')
    csvcache = '\n'.join(['{},{}'.format(t, cache[t])
        for t in sorted(cache.keys())])
    with open(BDC_CACHE_FILE, 'w') as o:
        o.write(csvcache)

class DocVector:
    """VSM"""
    def __init__(self, doc, label):
        terms = sorted(doc.split(' '))
        self.terms = sorted(list(frozenset(terms)))
        self.tf = [terms.count(t) for t in self.terms] # term frequency
        self.label = label

    def get_tf(self, term):
        """term frequency"""
        if term not in self.terms:
            return 0

        return self.tf[self.terms.index(term)]

class Corpus:
    """immutable set of Doc Vectors"""
    def __init__(self, docvecs):
        """preprocess for calculating bdc"""
        self.DOCVECS = docvecs
        self.CATEGORIES = None
        self.ALL_TERMS = None
        self.f_ci_cache = None
        self.bdc_cache = None
        self.select_by_category_cache = None
        self.bdc_cache = load_bdc_cache()

    def terms(self):
        """{t_i}"""
        if not self.ALL_TERMS:
            self.ALL_TERMS = set()

            for d in self.DOCVECS:
                for t in d.terms:
                    self.ALL_TERMS.add(t)

            self.ALL_TERMS = sorted(list(self.ALL_TERMS))

        return self.ALL_TERMS

    def categories(self):
        """{c_i}"""
        if not self.CATEGORIES:
            self.CATEGORIES = list(frozenset(sorted(set(
                [docvec.label for docvec in self.DOCVECS]))))

        return self.CATEGORIES

    def f_ci(self, c):
        """f(c_i) = frequency sum of all terms in category c_i"""
        if not self.f_ci_cache:
            self.f_ci_cache = [None] * len(self.categories())

        i = self.categories().index(c)

        if not self.f_ci_cache[i]:
            self.f_ci_cache[i] = sum([sum(d.tf)
                for d in self.select_by_category(c)])

        return self.f_ci_cache[self.categories().index(c)]

    def select_by_category(self, category):
        if not self.select_by_category_cache:
            self.select_by_category_cache = [
                [d for d in self.DOCVECS if c == d.label]
                for c in self.categories()
            ]

        return self.select_by_category_cache[self.categories().index(category)]

    def f_t_ci(self, term, category):
        """f(t, c_i) = frequency of term t in category c_i"""
        return sum([d.get_tf(term) for d in self.select_by_category(category)])

    def p_t_ci(self, term, category):
        """p(t|c_i) = f(t, c_i) / f(c_i)"""
        return self.f_t_ci(term, category) / self.f_ci(category)

    def sum_p_t_ci(self, term):
        """\sum_{i=1}{|C|} p(t|c_i)"""
        return sum([self.p_t_ci(term, c) for c in self.categories()])

    def G_t_ci(self, term, category):
        """G(t, c_i) = p(t|c_i)/(\sum_{i=1}{|C|} p(t|c_i))"""
        return self.p_t_ci(term, category) / self.sum_p_t_ci(term)

    def F_t_ci(self, term, category):
        """F(t, c_i) = G(t, c_i) log G(t, c_i)"""
        return x_logx(self.G_t_ci(term, category))

    def BH_t(self, term):
        """BH(t) = -\sum{i=1}{|C|} F(t, c_i)"""
        return -sum([self.F_t_ci(term, c) for c in self.categories()])

    def bdc(self, term):
        """bdc(t) = 1 - BH(t)/log(|C|)"""
        # logging.info('Calculating bdc for {}'.format(term))

        if not self.bdc_cache:
            self.bdc_cache = {}

        if term not in self.bdc_cache:
            abs_C = len(self.categories())
            self.bdc_cache[term] = 1 - self.BH_t(term)/math.log2(abs_C)

        b = self.bdc_cache[term]
        # logging.info('bdc({}): {}'.format(term, b))
        return b

    def predict_with_knn(self, knn_k_value, d):
        """
        return d's predicted label
        """
        logging.info('Predicting'.format())

        # remove terms that in the test-corpus however not in train-corpus
        whitelst = [t for t in d.terms if t in self.terms()]

        dw = [d.get_tf(t)*self.bdc(t) for t in whitelst] # weighted copy

        write_bdc_cache(self.bdc_cache)

        """
        weighted vectors of train data
        0. strip unused terms. only respect those terms occurs in
            docvec_to_predict
        1. weight with bdc
        """
        twv = [[doc.get_tf(t)*self.bdc(t) for t in whitelst]
            for doc in self.DOCVECS]

        write_bdc_cache(self.bdc_cache)

        labels = [d.label for d in self.DOCVECS]

        return knn_classify(5, dw, twv, labels)

def read_csv(csvfilename):
    """note: yield"""
    with open(csvfilename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            yield tuple(list(row))

def csv2corpus(csvfilename):
    """helper function"""
    return Corpus([DocVector(doc, label)
        for label, doc in read_csv(csvfilename)])

def x_logx(x):
    """
    special case for \lim_{i->0} 0logi -> 0, informally 0log0 = 0
    """
    if x != 0: return x*math.log2(x)
    return 0

def distance_to(v1, v2):
    """todo: Cosine simplify?"""
    return math.sqrt(sum([(x1-x2)*(x1-x2) for x1, x2 in zip(v1, v2)]))

def get_majority(votes):
    """get_majority([1, 2, 2, 1, 2, 3]) -> 2"""
    vote_dict = {label: votes.count(label) for label in votes}
    labels = list(vote_dict.keys())
    votes = list(vote_dict.values())
    return labels[votes.index(max(votes))]

def get_k_nearest_neighbors(knn_k_value, v, vx, labels):
    """
    vx for train: {[label, vec]}
    todo: divide-and-conquer optimize (see Introduction to Algorithms)
    """
    logging.info('Calculating distances ...')
    distances = list(enumerate([distance_to(v, vv) for vv in vx]))

    logging.info('Sorting distances ...')
    sorted_neighbors = [i
        for [i, _] in sorted(distances, key=operator.itemgetter(1))]

    return sorted_neighbors[:knn_k_value]

def knn_classify(knn_k_value, v, vx, labels):
    """
    note: preprocessed before
    """
    neighbors = get_k_nearest_neighbors(knn_k_value, v, vx, labels)

    return get_majority([labels[i] for i in neighbors])

def exam(testcorpus, traincorpus):
    """return correct_cnt, all_question_cnt"""
    correct_cnt, quiz_sum = 0, 0

    for d in testcorpus.DOCVECS:
        quiz_sum += 1
        logging.info('Quiz #{}'.format(quiz_sum))

        if traincorpus.predict_with_knn(5, d) == d.label:
            correct_cnt += 1
            logging.info('Correct: {}'.format(correct_cnt))

    return correct_cnt, quiz_sum

if __name__ == '__main__':
    """
    exam with TRAINCSV and TESTCSV, splited according to the LEWISSPLIT
    attribution in <NEUTERS> of the original .sgm file
    """
    logging.basicConfig(level=logging.INFO)
    TRAINCSV = 'train.csv'
    TESTCSV = 'test.csv'

    traincorpus = csv2corpus(TRAINCSV)
    testcorpus = csv2corpus(TESTCSV)

    print('ACCURACY: {}'.format(operator.truediv(*exam(testcorpus, traincorpus))))
