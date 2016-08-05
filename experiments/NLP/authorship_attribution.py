import sys
sys.path.append('../..')

import time, json, argparse, os.path
import nltk
from glob import glob
from gensim.models.word2vec import Word2Vec
from maxdiv import eval
import maxdiv_tools
from textutils import *


DATA_DIR = '../../../../datasets/pan16-author-diarization-training-dataset-problem-a/'


def loadText(filename):
    
    # Load and tokenize text
    with open('{}{}.txt'.format(DATA_DIR, filename)) as f:
        raw_text = f.read()
        text = nltk.word_tokenize(raw_text)
        text_sent = []
        for l in raw_text.split('\n'):
            if l.strip() != '':
                text_sent += [nltk.word_tokenize(s) for s in nltk.sent_tokenize(l)]
    
    # Load ground-truth
    with open('{}{}.truth'.format(DATA_DIR, filename)) as f:
        truth = json.load(f)['authors'][1]
        # Convert character positions to word positions
        for i in range(len(truth)):
            ca, cb = truth[i]['from'], truth[i]['to']
            wa = len(nltk.word_tokenize(raw_text[:ca]))
            wb = wa + len(nltk.word_tokenize(raw_text[ca:cb+1]))
            truth[i] = (wa, wb)
    
    return text, truth, text_sent


def printTruth(text, truth):
    
    print('GROUND TRUTH')
    for a, b in truth:
        print('\n{} - {}\n{}'.format(a, b, ' '.join(text[a:b])))


if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--files', help='problem file', nargs = '+', default = [])
    parser.add_argument('--feat', help='features', choices = ['word2vec', 'function_words'], default = 'word2vec')
    parser.add_argument('--model', help='word2vec model', type = str, default = 'brown_50_sg.pickle')
    parser.add_argument('--wordlist', help='list of function words', type = str, default = 'function-words_122.txt')
    parser.add_argument('--method', help='maxdiv method', choices=maxdiv.get_available_methods(), default = 'gaussian_global_cov')
    parser.add_argument('--mode', help='Mode for KL divergence computation', choices=['OMEGA_I', 'SYM', 'I_OMEGA', 'TS', 'LAMBDA', 'IS_I_OMEGA', 'JSD'], default='TS')
    parser.add_argument('--kernel_sigma_sq', help='kernel sigma square hyperparameter for Parzen estimation', type=float, default=1.0)
    parser.add_argument('--extint_min_len', help='minimum length of the extreme interval', default=50, type=int)
    parser.add_argument('--extint_max_len', help='maximum length of the extreme interval', default=250, type=int)
    parser.add_argument('--num_intervals', help='number of intervals', default=5, type=int)
    parser.add_argument('--td_dim', help='Time-Delay Embedding Dimension', default=1, type=int)
    parser.add_argument('--td_lag', help='Time-Lag for Time-Delay Embedding', default=1, type=int)
    
    args = parser.parse_args()
    args_dict = vars(args)
    parameters = {parameter_name: args_dict[parameter_name] for parameter_name in maxdiv_tools.get_algorithm_parameters() if parameter_name in args_dict}
    if ('num_intervals' in parameters) and (parameters['num_intervals'] <= 0):
        parameters['num_intervals'] = None
    parameters['kernelparameters'] = { 'kernel_sigma_sq' : args.kernel_sigma_sq }
    
    files = args.files
    if len(files) == 0:
        files = [os.path.splitext(os.path.basename(file))[0] for file in glob(DATA_DIR + 'problem-*.txt')]
        
    detections = []
    gt = []
    for file in files:
        
        # Load problem
        text, truth, text_sent = loadText(file)
        truth = [(a, b) for a, b in truth if b - a >= args.extint_min_len]
        if len(truth) == 0:
            continue
        gt.append(truth)
        if len(files) == 1:
            printTruth(text, truth)
        
        # Extract features
        if args.feat == 'word2vec':
            feat = text2mat(text, Word2Vec.load(args.model))
        elif args.feat == 'function_words':
            words = loadFunctionWords(args.wordlist)
            feat = wordFreq(text_sent, words)
        
        # Run detector
        start = time.time()
        intervals = maxdiv.maxdiv(feat, **parameters)
        stop = time.time()
        if args.feat == 'function_words':
            intervals = sentDet2wordDet(text_sent, intervals)
        detections.append(intervals)
        
        # Show results
        if len(files) == 1:
            printDetectedParagraphs(text, intervals)
            print('\nThe search for anomalous paragraphs in a text of {} words took {} seconds.'.format(len(text), stop - start))
    
    if len(files) > 1:
        ap = eval.average_precision(gt, detections, plot = True)
        print('AP: {}'.format(ap))