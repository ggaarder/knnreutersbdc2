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

def read_csv(csvfilename):
    """note: yield"""
    with open(csvfilename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            yield tuple(list(row))

def x_logx(x):
    """\lim_{i->0} 0logi -> 0, i.e. 0log0 = 0"""
    if x != 0: return x*math.log2(x)
    return 0

def get_majority(votes):
    """get_majority([1, 2, 2, 1, 2]) -> 2"""
    vote_dict = {label: votes.count(label) for label in votes}
    labels = list(vote_dict.keys())
    votes = list(vote_dict.values())
    return labels[votes.index(max(votes))]

def csv2traindat(csvfilename):
    """csv -> Traindat"""
    labels = []
    docvecs = []

    for label, doc in read_csv(csvfilename):
        labels.append(label)
        docvecs.append(Docvec(doc))

    return Traindat(labels, docvecs)

def get_k_nearest_neighbors(knn_k_value, vec_to_classify, traindat):
    """
    todo: divide-and-conquer optimize (see Introduction to Algorithms)
    """
    distances = [[i, vec_to_classify.distance_to(traindat[i][1])]
        for i in range(len(traindat))]

    sorted_neighbors = [traindat[i]
        for [i, _] in sorted(distances, key=operator.itemgetter(1))]

    return sorted_neighbors[:knn_k_value]

def knn_classify(knn_k_value, vec_to_classify, traindat):
    """note: preprocessed before"""
    neighbors = get_k_nearest_neighbors(knn_k_value, vec_to_classify,
        traindat)

    return get_majority([neighbor[0] for neighbor in neighbors])

class Docvec:
    """BOW"""
    def __init__(self, doc):
        if isinstance(doc, str):
            """bow will be sorted in key since words is sorted"""
            words = sorted(doc.split(' '))
            self.bow = {word: words.count(word) for word in words}
        else isinstance(doc, dict):
            self.bow = doc

    def get_term_frequency(self, term):
        """
        term frequency
        return a term's count of a docvec
        """
        return self.bow.get(term, 0)

    def get_sorted_terms(self):
        """
        not necessary to sort again since bow.keys() are sorted in __init___
        """
        return frozenset(self.bow.keys())

    def get_vec(self):
        """return the vector"""
        return list(self.bow.values())

    def mult_weight(self, weight):
        """multed_i = vec_i * w_i"""
        if len(weight) != len(self.bow):
            raise IndexError('len(weight) != len(bow)')

        for i, key in enumerate(dict):
            dict[key] *= weight[i]

    def weight_term(self, term, weight):
        """weight a certain term"""
        if term in self.bow:
            self.bow[term] *= weight

    def distance_to(self, v):
        """
        todo: Cosine simplify?
        """
        return math.sqrt(sum([(x1-x2)*(x1-x2)
            for x1, x2 in zip(self.get_vec(), v.get_vec())]))

class Traindat:
    """{topic + docvec}"""
    def __init__(self, labels, docvecs):
        """
        preprocess bdc
        todo: tf, idf
        """
        self.bdc_cache = {}
        self.DATS = zip(labels, docvecs) # note: immutable

        # {t_i}
        self.ALL_TERMS = frozenset(itertools.chain(
            set([docvec.get_sorted_terms() for [label, docvec] in self.DATS])))

        # {c_i}
        self.CATEGORIES = frozenset(
            sorted(set([label for [label, docvec] in self.DATS])))

        # |C| = the number of categories
        self.ABS_C = len(self.CATEGORIES)

        # f(c_i) = frequency sum of all terms in category c_i

        self.F_CI = {
            category: sum([
                len(docvec.get_sorted_terms())
                for [label, docvec] in self.select_dat_by_category(category)
            ])
            for category in self.CATEGORIES
        }

    def select_dat_by_category(self, category):
        """helper function"""
        return [
            [label, c] for [label, c] in self.dats
            if c == category
        ]

    def calc_f_t_ci(self, term, category):
        """f(t, c_i) = frequency of term t in category c_i"""
        return sum([docvec.get_term_frequency(term)
            for [label, docvec] in self.select_dat_by_category(category)])

    def calc_p_t_ci(self, term, category):
        """p(t|c_i) = f(t, c_i) / f(c_i)"""
        return self.calc_f_t_ci(term, category) / self.F_CI[category]

    def calc_sum_p_t_ci(self, term):
        """\sum_{i=1}{|C|} p(t|c_i)"""
        return sum([self.calc_p_t_ci(term, c) for c in self.CATEGORIES])

    def calc_G_t_ci(self, term, category):
        """G(t, c_i) = p(t|c_i)/(\sum_{i=1}{|C|} p(t|c_i))"""
        return self.calc_p_t_ci(term, category) / self.calc_sum_p_t_ci(term)

    def calc_F_t_ci(self, term, category):
        """F(t, c_i) = G(t, c_i) log G(t, c_i)"""
        return x_logx(self.calc_G_t_ci(term, category))

    def calc_BH_t(self, term):
        """BH(t) = -\sum{i=1}{|C|} F(t, c_i)"""
        return -sum([self.calc_F_t_ci(term, c) for c in self.CATEGORIES])

    def calc_bdc(self, term):
        """bdc(t) = 1 - BH(t)/log(|C|)"""
        if term not in self.bdc_cache:
            bdc_cache[term] = 1 - self.calc_BH_t(term)/math.log2(abs_C)

        return bdc_cache[term]

    def predict_with_knn(knn_k_value, docvec_to_predict):
        """
        return a predicted label
        """
        terms_whitelist = docvec_to_predict.get_sorted_terms()

        docvec_to_predict_weighted = copy.deep_copy(docvec_to_predict)

        """
        0. strip unused terms. only respect those terms occurs in
            docvec_to_predict
        1. weight with bdc
        """
        traindat_weighted = [
            [ label,
              { term: docvec.get_term_frequency(term) * self.calc_bdc(term)
                for term in terms_whitelist } ]
            for [label, docvec] in self.DATS]

        return knn_classify(5, docvec_to_predict_weighted, traindat_weighted)

def exam(testdat, traindat):
    """return correct_cnt, all_question_cnt"""
    correct_cnt, quiz_sum = 0, 0

    for [label, docvec] in testdat.DAT:
        quiz_sum += 1
        logging.info('Quiz #{}'.format(quiz_sum))

        if traindat.predict_with_knn(5, docvec) == topic:
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

    traindat = csv2traindat(TRAINCSV)
    testdat = csv2traindat(TESTCSV)

    print('ACCURACY: {}'.format(
        operator.div(*exam(testdat, traindat))))