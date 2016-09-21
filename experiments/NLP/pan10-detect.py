import sys
sys.path.append('../..')

import nltk
import os.path
from glob import glob
from gensim.models.word2vec import Word2Vec
from maxdiv.maxdiv import maxdiv
from textutils import text2mat


DIR_TRAIN = 'C:\\Users\\BJRN~1\\Downloads\\pan-plagiarism-corpus-2010\\suspicious-documents'
DIR_TEST  = 'C:\\Users\\BJRN~1\\Downloads\\pan-plagiarism-corpus-2010\\suspicious-documents-test'


def loadText(filename):
    
    # Load and tokenize text
    with open(filename, encoding = 'utf-8') as f:
        raw_text = f.read()
        text = nltk.word_tokenize(raw_text)
    
    return text


def findFiles(dir):

    files = glob(os.path.join(dir, '*.txt')) + glob(os.path.join(dir, '*', '*.txt'))
    return { os.path.basename(f) : f for f in files }


if __name__ == '__main__':
    
    # Parse arguments
    if len(sys.argv) < 3:
        print('Usage: {} <set = train|test> <detections-file> [<model = brown_50_sg.pickle>]'.format(sys.argv[0]))
        exit()
    
    dataset = sys.argv[1].lower()
    detFile = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else 'brown_50_sg.pickle'
    model = Word2Vec.load(model)
    
    # Find text files
    files = findFiles(DIR_TEST if dataset == 'test' else DIR_TRAIN)
    
    # Run detector and write results to a file
    with open(detFile, 'w') as outFile:
        for i, (fileId, file) in enumerate(files.items()):
            
            detections = maxdiv(text2mat(loadText(file), model), method = 'gaussian_cov', mode = 'TS',
                                num_intervals = 1000, extint_min_len = 100, extint_max_len = 5000,
                                proposals = 'hotellings_t')
            
            for a, b, score in detections:
                outFile.write('{},{},{},{}\n'.format(fileId, a, b, score))
            
            print('{}/{}'.format(i+1, len(files)))
            sys.stdout.flush()