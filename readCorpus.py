
"""Reads labels and sentences from the file. Each row represents a sentence. 
   The first index in each row is the length of the sentence, 
   and the rest are indexes of the words. Lables has the information about the words.
   A cheating word is added in the beginning, so the indexes actually represent the word number in the sentence.
   This function returns the matrix of sentences and the lables vector.
   Because sentence length is different, a maximum sentence length is set in the beginning
   and then the remaining indexes of the row are filled with -1s.
"""

import numpy as np
import io
import os
import collections
import logging
import hashlib
from six.moves import cPickle
import re

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


def readCorpus(datadir,minLength):

    corpusPath = os.path.join(datadir,'corpus')
    wordlistPath = os.path.join('data','wordlist')
    text_indexPath = os.path.join('data','corpus_index')

    if not (os.path.exists(wordlistPath)) or not (os.path.exists(text_indexPath)):
        logging.info('loading the raw corpus from %s'%(corpusPath))
        wordlist, text_index = processRawCorpus(corpusPath,minLength)
        with open(wordlistPath, 'wb') as tf:
            cPickle.dump(wordlist,tf)

        with open(text_indexPath,'wb') as cf:
            cPickle.dump(text_index,cf)

        labels, matrix  = loadText(wordlistPath, text_indexPath)

    else:
        logging.info('loading wordlist and matrix from %s and %s'%(wordlistPath,text_indexPath))
        labels, matrix = loadText(wordlistPath, text_indexPath)

    return labels, matrix

def processRawCorpus(rawPath,minLength):
    # with open(rawPath, 'r',encoding='utf-8') as tf:
    #     raw = tf.read()
    raw = ''
    try:
        with open(rawPath, 'r',encoding='utf-8') as f:
            hasher = hashlib.sha256()  # Make empty hasher to update piecemeal
            while True:
                block = f.read(64 * (1 << 20))  # Read 64 MB at a time; big, but not memory busting
                raw += block
                if not block:  # Reached EOF
                    break
                hasher.update(block.encode('utf-8')).hexdigest()
    except:IOError

    #clean chinese text
    #raw_clean = clean_chstr(raw)
    raw_clean = raw


    tokens = raw_clean.split()
    word_counts = collections.Counter(tokens)
    wordlist = [x[0] for x in word_counts.most_common()]
    wordlist = list(sorted(wordlist))
    wordlist = ['theCheatingWord']+wordlist
    vocab = {x:i for i, x in enumerate(wordlist)}

    text_index = []
    sentences = raw_clean.split('\n')
    for s, sent in enumerate(sentences):
        if len(sent)> minLength:
            sent = sent.strip().split(' ')
            length = [len(sent)]
            sent_idx = list(map(vocab.get, sent))
            text_index.append(length+sent_idx)

    return wordlist, text_index

def loadText(vocabPath, text_index):
    with open(text_index,'rb') as tf:
        corpus = cPickle.load(tf)
    # index = 0
    # matrix = np.zeros([len(corpus), maxLength],dtype='int')
    # for line in corpus:
    #     parsed = np.copy(line)
    #     matrix[index] = np.hstack((parsed, np.array((maxLength - len(parsed)) * [-1])))
    #     index = index + 1
    matrix = np.copy(corpus)

    with open(vocabPath,'rb') as lf:
        labels = cPickle.load(lf)

    return labels, matrix


# labels, matrix = readCorpus('data',80)
# print(labels,matrix)


#clean text for Chinese
def clean_chstr(string):
    string = string.lower()
    string = re.sub(r'[^\u4e00-\u9fa50-9\na-z]'," ",string)#去除非中文，非数字，非换行符的字符
    # string = re.sub(r"[\.\！\/\_,<>$%^\*()\-:;\+\‘\’\“\”\"\'\?]+|[+——！，。？、~@#￥%……&*（）《》]+", " ", string)
    string = re.sub(r"\n","&&",string)
    # string = re.sub(r"[a-zA-Z]+", '', string)
    string = re.sub(r"[0-9]+",'N',string)
    string = re.sub(r"\s{2,}",' ',string)
    string = re.sub(r"&&", "\n", string)

    return string







