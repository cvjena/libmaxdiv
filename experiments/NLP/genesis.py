""" Transform each word in the Book Genesis into a feature vector using word2vec and runs the MDI algorithm on the book to search for anomalous paragraphs. """

import sys
sys.path.append('../..')

import time
import numpy as np
from nltk.corpus import genesis
from maxdiv.maxdiv import maxdiv

import textutils


MIN_LEN = 50
MAX_LEN = 500
NUM_INTERVALS = 10


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('Usage: {} <word2vec-model> [<min-len = {}> [<max-len = {}> [<num-intervals = {}>]]]'.format(sys.argv[0], MIN_LEN, MAX_LEN, NUM_INTERVALS))
        exit()
    
    model = sys.argv[1]
    minLen = int(sys.argv[2]) if len(sys.argv) > 2 else MIN_LEN
    maxLen = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_LEN
    numIntervals = int(sys.argv[4]) if len(sys.argv) > 4 else NUM_INTERVALS
    
    text = genesis.words(fileids = 'english-kjv.txt')
    feat = textutils.text2mat(text, model)
    
    start = time.time()
    intervals = maxdiv(feat, method = 'gaussian_cov', mode = 'TS',
                       extint_min_len = minLen, extint_max_len = maxLen, num_intervals = numIntervals)
    stop = time.time()
    print('The search for anomalous paragraphs in a text of {} words took {} seconds.'.format(len(text), stop - start))
    
    textutils.printDetectedParagraphs(text, intervals)
    