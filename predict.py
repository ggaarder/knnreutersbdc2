import logging

def read_csv(csvfilename):
    with open(csvfilename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            yield row[0], row[1]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    traincsv = 'train.csv'
    testcsv = 'test.csv'

    logging.info('Reading training data ...')
    traindat = [[topic, doc] for topic, doc in read_csv(traincsv)]

    logging.info('Examing ...')
    correct_cnt = 0
    quiz_sum = 0
    for topic, doc in read_csv(testcsv):
        quiz_sum += 1
        if predict(doc, traindat) == topic: