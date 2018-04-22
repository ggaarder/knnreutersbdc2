import codecs
import logging
import os
import re
import nltk
import itertools
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import LancasterStemmer
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

STOPWORDS = nltk.corpus.stopwords.words('english')
LMTZR = WordNetLemmatizer()
STEMMER = LancasterStemmer()

def tokenize(doc):
    out = []

    for i in [i.split('/') for i in nltk.word_tokenize(doc)
        if re.match(r'[a-z]', i)]:
        preprocess = [re.sub(r'[^a-z]', '', j) for j in i
            if j not in STOPWORDS and re.match(r'[a-z]', j)]
        out += [i for i in preprocess if len(i) > 2]

    return out

def clean_doc(doc):
    doc = doc.lower()
    tokens = [STEMMER.stem(LMTZR.lemmatize(i)) for i in tokenize(doc)
        if i not in STOPWORDS]
    return ' '.join(sorted([i for i in tokens if len(i) > 2]))

class ReuterErr(Exception):
    pass

def parse_one_reuter_soup(reuter_soup):
    split = reuter_soup['lewissplit']

    if split not in ['TRAIN', 'TEST']:
        raise ReuterErr('UNKNOWN SPLIT')

    topic = get_topic(reuter_soup)
    if topic == 'NO TOPICS' or topic == 'MULTI TOPICS':
        raise ReuterErr('NOT ONE TOPIC')

    doc = get_doc(reuter_soup)
    if not doc:
        raise ReuterErr('NO BODY')

    return split, topic, clean_doc(doc)

def parse_one_sgm(reutdat):
    soup = BeautifulSoup(reutdat, 'html.parser')
    reuters = soup.find_all('reuters')

    for i in range(len(reuters)):
        try:
            yield parse_one_reuter_soup(reuters[i])
        except ReuterErr:
            pass

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