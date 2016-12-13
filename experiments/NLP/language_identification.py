""" Creates a mixed-language text from the Europarl corpus and runs the MDI algorithm on it. """

import sys
sys.path.append('../..')

import time
import numpy as np
from nltk.corpus import comtrans
from maxdiv.maxdiv import maxdiv

np.random.seed(0)


MIN_LEN = 10
MAX_LEN = 50
NUM_FOREIGN = 5


def makeMixedText(minLen = MIN_LEN, maxLen = MAX_LEN, numForeign = NUM_FOREIGN):
    
    numSents = len(comtrans.sents(fileids = ['alignment-de-en.txt']))
    
    foreign = np.zeros(numSents, dtype = bool)
    gt = []
    while len(gt) < numForeign:
        a = np.random.randint(minLen, numSents - minLen)
        b = a + np.random.randint(minLen, maxLen + 1)
        if not foreign[(a-minLen):(b+minLen)].any():
            foreign[a:b] = True
            gt.append((a,b))
    
    return [s.words if foreign[i] else s.mots for i, s in enumerate(comtrans.aligned_sents(fileids = ['alignment-de-en.txt']))], gt


def printDetectedParagraph(text, interval):
    
    a, b, _ = interval
    ax, bx = max(0, a - 10), min(len(text), b + 10)
    for i in range(ax, bx):
        if (i == a) or (i == b):
            print('-------------------------')
        print(' '.join(text[i]))


def printDetectedParagraphs(text, intervals):
    
    for i, interval in enumerate(intervals):
        print('\nDETECTION #{} (Score: {})\n'.format(i+1, interval[2]))
        printDetectedParagraph(text, interval)


def sent_feat(s):
    
    f = np.zeros((27,))
    samples = 0
    characters = 0
    for w in s:
        if w.isalpha():
            f[0] += len(w)
            for c in w:
                ind = ord(c) - ord('a')
                if (ind >= 0) and (ind < 26):
                    f[ind + 1] += 1
                    characters += 1
            samples += 1
    
    if samples > 0:
        f[0] /= samples
    return f


def text2feat(text):
    
    return np.vstack(sent_feat(s) for s in text)


if __name__ == '__main__':
    
    minLen = int(sys.argv[1]) if len(sys.argv) > 1 else MIN_LEN
    maxLen = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_LEN
    numForeign = int(sys.argv[3]) if len(sys.argv) > 3 else NUM_FOREIGN
    
    text, gt = makeMixedText(minLen, maxLen, numForeign)
    
    feat = text2feat(text)
    
    start = time.time()
    intervals = maxdiv(feat.T, method = 'gaussian_global_cov', mode = 'TS',
                       extint_min_len = minLen, extint_max_len = maxLen, num_intervals = numForeign * 2)
    stop = time.time()
    print('The search for anomalous paragraphs in a text of {} sentences took {} seconds.'.format(len(text), stop - start))
    
    printDetectedParagraphs(text, intervals)
    