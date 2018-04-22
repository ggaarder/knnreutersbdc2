import csv
import math
import logging
from operator import itemgetter

def read_csv(csvfilename):
    with open(csvfilename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            yield row[0], row[1]

def generate_terms_lst(docvec):
    if len(docvec[0]) == 1:
        return sorted(set[[term for term in docvec]])
    elif len(docvec[0]) == 2:
        return sorted(set([term for [term, frequency] in docvec]))

def calc_term_frequency_sum_of_all_terms(docvec):
    return sum([frequency for [term, frequency] in docvec])

def calc_term_frequency(term, docvec):
    for t, frequency in docvec:
        if t == term:
            return frequency
    return 0

def x_logx(x):
    if x != 0: return x*math.log2(x)
    return 0

def div(a, b):
    """calc a/b"""
    if a == 0: return 0
    return a/b

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

    p_t_ci = [div(f_t_ci[i], f_ci[i]) for i in range(abs_C)]
    sum_p_t_ci = sum(p_t_ci)
    G_t_ci = [div(p_t_ci[i], sum_p_t_ci) for i in range(abs_C)]
    F_t_ci = [x_logx(G_t_ci[i]) for i in range(abs_C)]
    BH_t = -sum(F_t_ci)
    bdc = 1 - BH_t/math.log2(abs_C)

    return bdc

def get_majority(votes):
    vote_dict = {label: votes.count(label) for label in votes}
    labels = list(vote_dict.keys())
    votes = list(vote_dict.values())
    return labels[votes.index(max(votes))]

def mult_weight(vec, w):
    if len(vec) != len(w): raise OverflowError(
        'len(vec): {}, len(w): {}'.format(len(vec), len(w)))

    return [vec[i]*w[i] for i in range(vec)]

def strip_unused_terms_of_docvec(termslst, docvec):
    return [term_tf for term_tf in docvec if term_tf[0] in termslst]

def knn_preprocess(docvec_to_predict, traindat):
    """remove unused terms and weight terms of the doc vector"""
    terms = generate_terms_lst(docvec_to_predict)
    terms_bdc = [calc_bdc(i, traindat) for i in terms]

    docvec_to_predict_weighted = mult_weight(docvec_to_predict, terms_bdc)

    train_docvecs_stripped_unused_terms = [
        strip_unused_terms_of_docvec(terms, docvec)
        for [label, docvec] in traindat]

    train_docvecs_stripped_weighted = [
        mult_weight(docvec, terms_bdc)
        for docvec in train_docvecs_stripped_unused_terms]

    traindat_stripped_weighted = [
        [traindat[i][0], train_docvecs_stripped_weighted[i]]
        for i in range(traindat)]

    return docvec_to_predict_weighted, traindat_stripped_weighted


def predict_with_knn(knn_k_value, docvec_to_predict, traindat):
    preprocessed_docvec_to_predict, preprocessed_traindat = knn_preprocess(
        docvec_to_predict, traindat)

    neighbors = knn_get_k_nearest_neighbors(knn_k_value,
        preprocessed_docvec_to_predict, preprocessed_traindat)

    return get_majority([neighbor[0] for neighbor in neighbors])

def preprocess_csv(dat):
    """vectorize"""
    out = []

    for [topic, doc] in dat:
        words = sorted(doc.split(' '))
        vector = [[word, words.count(word)] for word in sorted(set(words))]
        out.append([topic, vector])

    return out

def exam(testdat, traindat):
    correct_cnt, quiz_sum = 0, 0

    for topic, docvec in testdat:
        quiz_sum += 1
        logging.info('Quiz #{}'.format(quiz_sum))

        if topic == predict_with_knn(docvec, traindat, knn_k_value=5):
            correct_cnt += 1
            logging.info('Correct: {}'.format(correct_cnt))

    return correct_cnt/quiz_sum

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    traincsv = 'train.csv'
    testcsv = 'test.csv'
    traindat = preprocess_csv(
        [[topic, doc] for topic, doc in read_csv(traincsv)])
    testdat = preprocess_csv(
        [[topic, doc] for topic, doc in read_csv(testcsv)])

    print('ACCURACY: {}'.format(exam(testdat, traindat)))