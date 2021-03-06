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
        self.f_t_ci_cache = None

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
        """
        f(t, c_i) = frequency of term t in category c_i

        cache because that according to profile:
           478276    0.926    0.000  101.256    0.000 predict.py:98(f_t_ci)
        cache structure is f_t_ci = f_t_ci_cache[i][t]
        """
        if not self.f_t_ci_cache:
            self.f_t_ci_cache = [None] * len(self.categories())

        i = self.categories().index(category)
        if not self.f_t_ci_cache[i]:
            self.f_t_ci_cache[i] = {}

        if term not in self.f_t_ci_cache[i]:
            r = sum([d.get_tf(term)
                for d in self.select_by_category(category)])
            self.f_t_ci_cache[i][term] = r

        return self.f_t_ci_cache[i][term]

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

def vector_distance(v1, v2):
    """
    to compare two vectors, using cosine_similarity instead of this can get
    better performance while the error is very little.

    see http://cs.carleton.edu/cs_comps/0910/netflixprize/final_results/knn/
    index.html
    """
    return math.sqrt(sum([(x1-x2)*(x1-x2) for x1, x2 in zip(v1, v2)]))

def inner_product(v1, v2):
    return sum([x1*x2 for x1, x2 in zip(v1, v2)])

def vector_abs(v, cache={}):
    v_hash = repr(v)

    if v_hash not in cache:
        cache[v_hash] = math.sqrt(sum([i*i for i in v]))

    return cache[v_hash]

def cosine_similarity(v1, v2): # accuracy ~ 0.16. very strange
    try:
        return inner_product(v1, v2)/vector_abs(v1)/vector_abs(v2)
    except ZeroDivisionError:
        v1x, v2x = v1+[1], v2+[1] # todo: is this workaround the best?
        return inner_product(v1x, v2x)/vector_abs(v1x)/vector_abs(v2x)

def vector_similarity(v1, v2):
    USE_COSINE = False

    if USE_COSINE:
        return cosine_similarity(v1, v2)
    else:
        return vector_distance(v1, v2)

def get_majority(votes):
    """get_majority([1, 2, 2, 1, 2, 3]) -> 2"""
    vote_dict = {label: votes.count(label) for label in votes}
    labels = list(vote_dict.keys())
    votes = list(vote_dict.values())
    return labels[votes.index(max(votes))]

def get_k_nearest_neighbors(knn_k_value, v, vx, labels):
    """
    Arguments *vx*, *labels* are train data
    Returns a list of **indexs** of the k-nearest neighbors
        indexs in *vx* and *labels*
    Assume len(vx) >= knn_k_value
    """
    # quicksort-like divide-and-conquer optimize to finding K-Nearest
    DC_OPT_TOGGLE = False

    logging.info('Calculating similarities')
    similarities = list(enumerate([vector_similarity(v, vv) for vv in vx]))
    SORT_KEY = operator.itemgetter(1)

    logging.info('Sorting similarities')

    if not DC_OPT_TOGGLE:
        sorted_neighbors = [i
            for [i, _] in sorted(similarities, key=SORT_KEY)]

        return sorted_neighbors[:knn_k_value]
    else:
        def get_k_smallest(k, xs, key=SORT_KEY):
            """
            WARNING: partition in-place
            quicksort-like divide-and-conquer optimize to finding K-Nearest
            """
            if len(xs) < k: raise OverflowError()
            elif len(xs) == k : return xs

            # partition xs to xs[0..i] < xs[i+1] < xs[i+2..]

            x = key(xs[-1]) # soldier
            # note: if write x = xs[-1], we will get a ref, and x will change
            # as xs[-1] may change (maybe) during the partition

            i = -1
            for j in range(0, len(xs)-1): # j from 0 to '-2', i.e. len(xs)-2
                if key(xs[j]) <= x: # see the note about key(xs[-1]) above
                    i += 1
                    xs[i], xs[j] = xs[j], xs[i]
            xs[i+1], xs[-1] = xs[-1], xs[i+1]

            # we need how many neighbors more other than xs[0..i+1]
            delta = k-(i+1)

            if delta > 0: # needs more
                return xs[:i+2] + get_k_smallest(k-delta, xs[i+2:i+2+delta])
                #      xs[0..i+1]                         ^^^^^^^^^^^^^^^^
                #                      xs[i+1:i+1+delta], *delta* elements
                return get_k_smallest(k, xs[:i+1])
            else:
                return xs[:k] # xs[0..k-1], k elements

        return [i
            for [i, _] in get_k_smallest(knn_k_value, similarities)]

def knn_classify(knn_k_value, v, vx, labels):
    """
    note: preprocessed before
    """
    neighbors = get_k_nearest_neighbors(knn_k_value, v, vx, labels)

    return get_majority([labels[i] for i in neighbors])

def test_classify(testcorpus, traincorpus):
    """return the predicted results list"""
    results = []
    correct_cnt = 0

    for i, d in enumerate(testcorpus.DOCVECS):
        results.append(traincorpus.predict_with_knn(5, d))

        if results[-1] == d.label:
            correct_cnt += 1

        logging.info('Quiz #{} (accuracy {})'.format(i, correct_cnt/(i+1)))

    return results

def simple_exam(results):
    """return correct_cnt, all_question_cnt"""
    correct_cnt = len([None for i, result in enumerate(results)
        if result == traincorpus.DOCVECS[i].label])
    quiz_sum = len(results)

    return correct_cnt, quiz_sum

def macro_micro_f1(results, correct_results):
    """
    see bdc [17]: Beyond tfidf weighting for text categorization in the vector
    space model.
    Pascal Soucy and Guy W Mineau. In IJCAI, volume 5, pages 1130–1135, 2005.
    """

    def f1(c):
        """
        bdc[17]:
        ---------------------------------------------------------------
                              Classifier      Classifier
                             positive label  negative label
        True positive label       A               B
        True negative label       C               D

        For any category, the classifier precision is defined as A/(A+C) and
        the recall as A/(A+B).

        F1 = (p+r)/(2*p*r)
        -----------------------------------------------------------------------"""
        def ABCD_cnt(c):
            """A, B, C, D for category c"""
            answers_and_keys = zip(results, correct_results)

            # True Positive and classifier Positive
            A_cnt = len([None
                for i, [resu, corr] in enumerate(answers_and_keys)
                if corr == c and resu == c ])

            # True Positive and Classifier Negative
            B_cnt = len([None
                for i, [resu, corr] in enumerate(answers_and_keys)
                if corr == c and resu != c ])

            # True Negative and Classifier Positive
            C_cnt = len([None
                for i, [resu, corr] in enumerate(answers_and_keys)
                if corr != c and resu == c ])

            # True Negative and Classifier Negative
            D_cnt = len([None
                for i, [resu, corr] in enumerate(answers_and_keys)
                if corr != c and resu != c ])

        def precison_recall(A_cnt, B_cnt, C_cnt, D_cnt):
            return A_cnt/(A_cnt+C_cnt), A_cnt/(A_cnt+B_cnt)

        p, r = precision_recall(*ABCD_cnt(c))

        return (p+r)/(2*p*r)

    if len(results) != len(correct_results): raise LookupError()

    categories = sorted(set(correct_results))
    f1s = [f1(c) for c in categories]
    macro_f1 = sum(f1s)/len(f1s)

    # todo: really in this way?
    # I don't know the exact formula. Just according to bdc[17], I got that
    # one below. It is correct?
    #
    # bdc[17]: the micro-F1 average weighs large categories more than smaller
    # ones
    f1s_weighted = [ f1s[categories.index(c)]*correct_results.count(c)
        for c in categories]
    micro_f1 = sum(f1s_weighted)/len(f1s_weighted)

    return macro_f1, micro_f1

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

    results = test_classify(testcorpus, traincorpus)
    print('ACCURACY: {}'.format(operator.truediv(*simple_exam(results))),
        'Macro, Micro F1: {}, {}',format(*macro_micro_f1(results,
            [d.label for d in testcorpus.DOCVECS])))
