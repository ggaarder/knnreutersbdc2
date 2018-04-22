import csv
import math
import logging
import operator

"""
Text Classification
Train/Test: Neuters 21578
Method: KNN, bdc
"""

def read_csv(csvfilename):
    """note: yield"""
    with open(csvfilename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            yield row[0], row[1]

def generate_terms_lst(docvec):
    """return a set"""
    if len(docvec[0]) == 1:
        return sorted(set[[term for term in docvec]])
    elif len(docvec[0]) == 2:
        return sorted(set([term for [term, frequency] in docvec]))

def calc_term_frequency_sum_of_all_terms(docvec):
    """return the term count of a docvec"""
    return sum([frequency for [term, frequency] in docvec])

def calc_term_frequency(term, docvec):
    """return a term's count of a docvec"""
    for t, frequency in docvec:
        if t == term:
            return frequency
    return 0

def x_logx(x):
    """\lim_{i->0} 0logi -> 0, i.e. 0log0 = 0"""
    if x != 0: return x*math.log2(x)
    return 0

def div(a, b):
    """
    calc a/b,
    we assume that 0/0 = 0 <------- TODO: Really???????????
    """
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
    """get_majority([1, 2, 2, 1, 2]) -> 2"""
    vote_dict = {label: votes.count(label) for label in votes}
    labels = list(vote_dict.keys())
    votes = list(vote_dict.values())
    return labels[votes.index(max(votes))]

def mult_weight(vec, w):
    """multed_i = vec_i * w_i"""
    if len(vec) == len(w):
        return list(itertools.starmap(operator.mul, zip(vec, w)))

def strip_unused_terms_of_docvec(termslst, docvec):
    """strip terms not in termslst"""
    return [term_tf for term_tf in docvec if term_tf[0] in termslst]

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

def calc_distance(a, b):
    """todo: Cosine simplify?"""
    return math.sqrt(sum([(x1-x2)*(x1-x2) for x1, x2 in zip(a, b)]))

def knn_get_k_nearest_neighbors(knn_k_value, testvec, traindat):
    """todo: optimize finding k-nearest with divide-and-conquer"""
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

def doc2vec(doc):
    """BOW"""
    words = sorted(doc.split(' '))
    words_cnt = {word: words.count(word) for word in words}

    return zip(words_cnt.keys(), words_cnt.values())

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