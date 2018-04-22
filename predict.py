import csv
import math
import logging
import operator

"""
Text Classification
Train/Test: Neuters 21578
Method: KNN, bdc
todo: SVM, tf*bdc, idf, etc.
"""

class docvec:
    """BOW"""
    def __init__(self, doc):
        """bow will be sorted in key since words is sorted"""
        words = sorted(doc.split(' '))
        self.bow = {word: words.count(word) for word in words}

    def get_tf(self, term):
        """
        term frequency
        return a term's count of a docvec
        """
        return self.bow[term]

    def get_terms(self):
        """return a set"""
        return self.bow.keys()

    def get_vec(self):
        """return the vector"""
        return list(self.bow.values())

    def mult_weight(self, vec, weight):
        """multed_i = vec_i * w_i"""
        if len(weight) != len(self.bow):
            raise IndexError('len(weight) != len(bow)')

        for i, key in enumerate(dict):
            dict[key] *= weight[i]

    def strip_unused_terms(self, whitelist):
        """strip terms not in whitelist"""
        self.bow = {term: self.bow[term] for term in self.bow
            if term in whitelist}

    def distance_to(self, v):
        """
        todo: Cosine simplify?
        """
        return math.sqrt(sum([(x1-x2)*(x1-x2)
            for x1, x2 in zip(self.get_vec(), v.get_vec())]))

class traindat:
    """list of docvec"""

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

def calc_bdc(term, traindat):
    """
    bdc(t) = 1 - BH(t)/log(|C|)
    |C| = the number of categories
    BH(t) = -\sum{i=1}{|C|} F(t, c_i)
    F(t, c_i) = G(t, c_i) log G(t, c_i)
    G(t, c_i) = p(t|c_i)/(\sum_{i=1}{|C|} p(t|c_i))
    p(t|c_i) = f(t, c_i) / f(c_i)
    f(c_i) = frequency sum of all terms in category c_i
    f(t, c_i) = frequency of term t in category c_i
    """
    categories = sorted(set([topic for [topic, docvec] in traindat]))
    abs_C = len(categories)

    docvecs_of_each_category = [
        [docvec for category, docvec in traindat if category == c]
        for c in categories]

    f_ci = [
        sum([calc_term_frequency_sum_of_all_terms(docvec)
            for docvec in docvecs_in_this_category])
        for docvecs_in_this_category in docvecs_of_each_category]

    f_t_ci = [
        sum([calc_term_frequency(term, docvec)
            for docvec in docvecs_in_this_category])
        for docvecs_in_this_category in docvecs_of_each_category]

    p_t_ci = [f_t_ci[i]/f_ci[i] for i in range(abs_C)]
    sum_p_t_ci = sum(p_t_ci)
    G_t_ci = [p_t_ci[i]/sum_p_t_ci for i in range(abs_C)]
    F_t_ci = [x_logx(G_t_ci[i]) for i in range(abs_C)]
    BH_t = -sum(F_t_ci)
    bdc = 1 - BH_t/math.log2(abs_C)

    return bdc

def get_majority(votes):
    """get_majority([1, 2, 2, 1, 2]) -> 2"""
    vote_dict = {label: votes.count(label) for label in votes}
    labels = list(vote_dict.keys())
    votes = list(vote_dict.values())
    return labels[votes.index(max(votes))]

def knn_preprocess(docvec_to_predict, traindat):
    """remove unused terms and weight terms of the doc vector"""
    terms = generate_terms_lst(docvec_to_predict)
    terms_bdc = [calc_bdc(i, traindat) for i in terms]

    return mult_weight(docvec_to_predict, terms_bdc), [
        [
            label,
            mult_weight(
                strip_unused_terms_of_docvec(terms, docvec),
                terms_bdc)
        ] for [label, docvec] in traindat
    ]

def knn_get_k_nearest_neighbors(knn_k_value, testvec, traindat):
    """
    todo: divide-and-conquer optimize (see Introduction to Algorithms)
    """
    numbered_distances = [[i, calc_distance(testvec, traindat[i][1])]
        for i in range(len(traindat))]

    k_nearest_neighbors = [traindat[i]
        for [i, _] in sorted(numbered_distances, key=operator.itemgetter(1))]

    return k_nearest_neighbors[:knn_k_value]

def predict_with_knn(knn_k_value, docvec_to_predict, traindat):
    """return a predicted topic"""
    neighbors = knn_get_k_nearest_neighbors(knn_k_value,
        *knn_preprocess(docvec_to_predict, traindat))

    return get_majority([neighbor[0] for neighbor in neighbors])

def exam(testdat, traindat):
    """return correct_cnt, all_question_cnt"""
    correct_cnt, quiz_sum = 0, 0

    for topic, docvec in testdat:
        quiz_sum += 1
        logging.info('Quiz #{}'.format(quiz_sum))

        if topic == predict_with_knn(5, docvec, traindat):
            correct_cnt += 1
            logging.info('Correct: {}'.format(correct_cnt))

    return correct_cnt, quiz_sum

if __name__ == '__main__':
    """
    exam with TRAINCSV and TESTCSV, splited according to the LEWISSPLIT
    attribution in <NEUTERS> in the original .sgm file
    """
    logging.basicConfig(level=logging.INFO)
    TRAINCSV = 'train.csv'
    TESTCSV = 'test.csv'

    traindat = [[topic, doc2vec(doc)] for topic, doc in read_csv(TRAINCSV)]
    testdat = [[topic, doc2vec(doc)] for topic, doc in read_csv(TESTCSV)]

    print('ACCURACY: {}'.format(
        div(*exam(testdat, traindat))))