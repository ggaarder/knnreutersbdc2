import logging
import os
import re
import json
import pickle
import itertools
from gzip import GzipFile
from bs4 import BeautifulSoup as bs
import numpy as np
import nltk

OUT_JSON = 'preprocessed.json'
REUTERS_TGZ = '/home/ggaarder/home/download/websites/www.daviddlewis.com\
/resources/testcollections/reuters21578/reuters21578.tar.gz'
STOPWORDS = nltk.corpus.stopwords.words('english')
LMTZR = nltk.stem.wordnet.WordNetLemmatizer()
STEMMER = nltk.stem.LancasterStemmer()
TOPICS_WHITELIST = [ 'earn', 'acq', 'trade', 'ship', # see the bdc paper
                     'grain', 'crude', 'interest', 'money-fx']
RARE_TERMS_LINE = 100 # terms appears less than ___ times will be ignored
STAGE1_PICKLE = 'preprocessed.stage1.pickle'
STAGE2_PICKLE = 'preprocessed.stage2.pickle'

def parse(news_soup):
    if len(news_soup.topics.find_all('d')) != 1 or news_soup.topics.d.string not in TOPICS_WHITELIST:
        raise
            
    news = {
        'topic': news_soup.topics.d.string,
        'oldid': news_soup['oldid'],
        'newid': news_soup['newid'],
        'lewissplit': news_soup['lewissplit'],
        'cgisplit': news_soup['cgisplit'],
    }

    doc = news_soup.title.string+' '+news_soup.body.string
    doc = [re.sub(r'[^a-z]', '', t) for t in nltk.word_tokenize(doc.lower())]
    doc = [STEMMER.stem(LMTZR.lemmatize(t)) for t in doc
           if len(t)>2 and t not in STOPWORDS]
    news['tf'] = {t: doc.count(t) for t in sorted(set(doc))}

    return news

def stage1():
    with GzipFile(REUTERS_TGZ) as gz:
        raw_SGML = str(gz.read())
    all_terms = set()
    all_topics = sorted(TOPICS_WHITELIST)
    all_news = []
    
    for i in itertools.count(1):
        logging.info('%d', i)
        m = re.match(r'.+?(<REUTERS.+?</REUTERS>)', raw_SGML)
        if not m:
            break
        raw_SGML = raw_SGML[m.end():]
        try:
            news_soup = bs(m.groups()[0], 'html.parser').reuters
            news = parse(news_soup)
        except:
            continue
        all_terms |= frozenset(news['tf'].keys())
        all_news.append(news)

    with open(STAGE1_PICKLE, 'wb') as o:
        pickle.dump([all_terms, all_topics, all_news], o)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(STAGE1_PICKLE):
        stage1()
    with open(STAGE1_PICKLE, 'rb') as r:
        all_terms, all_topics, all_news = pickle.load(r)
    if not os.path.exists(STAGE2_PICKLE):
        # remove rare terms
        all_terms = [t for t in sorted(all_terms)
                     if sum([news['tf'].get(t, 0)
                             for news in all_news]) > RARE_TERMS_LINE]
        with open(STAGE2_PICKLE, 'wb') as o:
            pickle.dump([all_terms, all_topics, all_news], o)
    with open(STAGE2_PICKLE, 'rb') as r:
        all_terms, all_topics, all_news = pickle.load(r)
        
    for news in all_news:
        terms = list(news['tf'].keys())
        for t in terms:
            if t not in all_terms:
                del news['tf'][t]
    all_news = [news for news in all_news
                if news['tf'].keys() != 0]
                
    out_json = {
        'all_terms': all_terms,
        'all_topics': all_topics,
        'news': all_news
    }
                    
    with open(OUT_JSON, 'w') as o:
        json.dump(out_json, o, indent=2)
