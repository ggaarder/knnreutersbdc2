import csv
import logging

def read_csv(csvfilename):
    with open(csvfilename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            yield row[0], row[1]

def wordlst_gen(docs):
    words = set()

    for doc in docs:
        for word in doc.split(' '):
            words.add(word)

    return words

def doc2vec(doc, wordlst):
    """todo: bdc"""
    doc_words = doc.split(' ')
    return [doc_words.count(i) for i in wordlst]

def predict(docs, traindat):
    """
    Note: If predict(doc, traindat), traindat's word list will be generated every call. time-consuming

    Using KNN, with VSM+bdf
    """
    words = wordlst_gen(docs + [i[1] for i in traindat])
    return 'xxx'

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    traincsv = 'train.csv'
    testcsv = 'test.csv'

    logging.info('Reading training data ...')
    traindat = [[topic, doc] for topic, doc in read_csv(traincsv)]

    logging.info('Reading test data ...')
    questions, correct_answers = [], []

    for topic, doc in read_csv(testcsv):
        questions.append(doc)
        correct_answers.append(topic)

    logging.info('Examing ...')
    answers = predict(questions, traindat)
    correct_cnt = len(
        [i for i, ans in enumerate(answers) if ans == correct_answers[i]])
    quiz_sum = len(answers)

    print('ACCURACY: {}'.format(correct_cnt/quiz_sum))