import codecs
import logging
import os
import re
from bs4 import BeautifulSoup

def get_topic(reuter):
    if not reuter.topics or not reuter.topics.d:
        return 'NO TOPICS'
    elif 1 != len(reuter.topics.find_all('d')):
        return 'MULTI TOPICS'
    else:
        return reuter.topics.d.string

def get_doc(reuter):
    if reuter.text and reuter.title and reuter.body:
        title = reuter.title.string
        body = reuter.body.string

        return '{}\t{}'.format(title, body)

def clean_doc(doc):
    def conv_abbr(doc):
        """
        U.K. U.S.A. P.R.C. etc.. -> UK USA PRC ...
        if not -> U K       U S A         P R C      (since [^A-Z] -> ' ')
        """
        result = []
        for i in doc.split(' '):
            if re.match(r'^([A-Z]\.?)+$', i):
                i = i.replace('.', '')
            result.append(i)

        return ' '.join(result)

    doc = doc.upper()
    doc = re.sub(r'[\'"]+', '', doc)
    doc = conv_abbr(doc)
    doc = re.sub(r'[^A-Z]+', ' ', doc)
    doc = re.sub(r'\s+', ' ', doc)
    doc = re.sub(r'^\s+', '', doc)
    doc = re.sub(r'\s$', '', doc)
    return doc.lower()

def parse_one_sgm(reutdat):
    soup = BeautifulSoup(reutdat, 'html.parser')
    reuters = soup.find_all('reuters')

    for i in range(len(reuters)):
        reuter = reuters[i]
        split = reuter['lewissplit']

        if split not in ['TRAIN', 'TEST']:
            continue

        topic = get_topic(reuter)
        if topic == 'NO TOPICS' or topic == 'MULTI TOPICS':
            continue

        doc = get_doc(reuter)
        if not doc:
            continue

        yield split, topic, clean_doc(doc)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    with open('train.csv', 'w') as trainout, open('test.csv', 'w') as testout:
        for i in range(0, 22):
            logging.info('Parsing {}'.format(i))

            sgmfilename = r'C:\Users\01\Desktop\reu\reut2-{:03d}.sgm'.format(i)
            with codecs.open(sgmfilename, encoding='utf-8',
                errors='backslashreplace') as f:
                reutdat = f.read()

            for lewissplit, topic, doc in parse_one_sgm(reutdat):
                csvline = '{},{}\n'.format(topic, doc)

                if lewissplit == 'TRAIN':
                    trainout.write(csvline)
                elif lewissplit == 'TEST':
                    testout.write(csvline)
                elif lewissplit != 'NOT-USED':
                    logging.warning('Unknown lewissplit {}'.format(lewissplit))