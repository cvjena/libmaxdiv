import sys
sys.path.append('../..')

import numpy as np
import nltk
from gensim.models.word2vec import Word2Vec
from maxdiv import maxdiv


def printDetectedParagraph(text, interval):
    
    a, b, _ = interval
    ax, bx = max(0, a - 50), min(len(text), b + 50)
    print(' '.join(text[ax:a]))
    print('-------------------------')
    print(' '.join(text[a:b]))
    print('-------------------------')
    print(' '.join(text[b:bx]))


def printDetectedParagraphs(text, intervals):
    
    for i, interval in enumerate(intervals):
        print('\nDETECTION #{}: {} - {} (Score: {})\n'.format(i+1, *interval))
        printDetectedParagraph(text, interval)


def loadFunctionWords(file):
    
    with open(file) as f:
        words = [w.strip().lower() for w in f if w.strip() != '']
    return set(words)


def text2mat(text, model):
    
    if not isinstance(model, Word2Vec):
        model = Word2Vec.load(model)
    
    feat = np.zeros((model.vector_size, len(text)))
    for i, w in enumerate(text):
        if w.lower() in model:
            feat[:,i] = model[w.lower()]
    return feat


def wordFreq(text, words):
    
    word_ids = { w : i for i, w in enumerate(words) }
    
    feat = np.zeros((len(words), len(text)))
    for i, sent in enumerate(text):
        for w in sent:
            numWords = 0
            if w.lower() in words:
                feat[word_ids[w.lower()], i] += 1
            if w.isalpha():
                numWords += 1
            if numWords > 0:
                feat[:, i] /= numWords
    return feat


def sentDet2wordDet(sents, intervals):
    
    det = []
    for a, b, score in intervals:
        wa = sum(len(s) for s in sents[:a])
        wb = wa + sum(len(s) for s in sents[a:b])
        det.append((wa, wb, score))
    return det