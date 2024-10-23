import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

MAX_NUM_WORD = 2000
EMBEDDING_SIZE = 100
num_classes = 91

def read_vector(filename):
    wordVectors = []
    vocab = []
    fileObject = open(filename, 'r')
    for i, line in enumerate(fileObject):
        if i==0 or i==1: # first line is a number (vocab size)
            continue
        line = line.strip()
        word = line.split()[0]
        vocab.append(word)
        wv_i = []
        for j, vecVal in enumerate(line.split()[1:]):
            wv_i.append(float(vecVal))
        wordVectors.append(wv_i)
    wordVectors = np.asarray(wordVectors)
    vocab_dict = dict(zip(vocab, range(1, len(vocab)+1))) # no 0 id; saved for padding
    print("Vectors read from: "+filename)
    return wordVectors, vocab_dict, vocab

def read_corpus(x_path, y_path, id_path, vocab_dict, reduced_vocab, num_docs):
    y = np.load(y_path)
    docs = []
    ids = np.load(id_path)
    X = np.zeros((len(ids), MAX_NUM_WORD), dtype=np.int32)
    print("number of data: ", len(ids))
    for i, ID in enumerate(ids):
        with open(x_path+'/'+ID) as r_f:
            text = r_f.read() 
            doc = [vocab_dict[w] for w in text.split() if w in vocab_dict]
            X[i, :len(doc)] = doc
            docs.append(text.strip())
    count_vect = CountVectorizer(vocabulary=reduced_vocab)
    bow = count_vect.transform(docs).toarray()
    return X, y, bow

def load_data(topic_vocab_size=2000):
    wv_matrix, vocab_dict, vocab = read_vector("processed/embedding/WordVec.txt")
    vocab_size = len(vocab)
    print("original vocab size: ", vocab_size)

    reduced_vocab = []
    with open("processed/topic_model_vocab_{}.txt".format(topic_vocab_size)) as r_f:
        for line in r_f:
            reduced_vocab.append(line.strip())
    
    train_x_rnn, train_y, train_x_bow = read_corpus("processed/text", 
                'processed/train_y.npy', 'processed/train_ids.npy', vocab_dict, reduced_vocab, 11550)
    
    test_x_rnn, test_y, test_x_bow = read_corpus("processed/text", 
                'processed/test_y.npy', 'processed/test_ids.npy', vocab_dict, reduced_vocab, 8141)

    return wv_matrix, vocab_dict, vocab, reduced_vocab, train_x_rnn, train_y, train_x_bow, test_x_rnn, test_y, test_x_bow

def load_data_by_batch(batch_ids, vocab_dict, reduced_vocab, x_path='processed/text'):
    docs = []
    X = np.zeros((len(batch_ids), MAX_NUM_WORD), dtype=np.int32)
    for i, ID in enumerate(batch_ids):
        with open(x_path+'/'+ID) as r_f:
            text = r_f.read() 
            doc = [vocab_dict[w] for w in text.split() if w in vocab_dict]
            X[i, :len(doc)] = doc
            docs.append(text.strip())
    count_vect = CountVectorizer(vocabulary=reduced_vocab)
    bow = count_vect.transform(docs).toarray()
    return X, bow

def load_bow_by_batch(batch_ids, reduced_vocab, x_path='processed/text'):
    docs = []
    for ID in batch_ids:
        with open(x_path+'/'+ID) as r_f:
            text = r_f.read() 
            docs.append(text.strip())
    count_vect = CountVectorizer(vocabulary=reduced_vocab)
    bow = count_vect.transform(docs).toarray()
    return bow

def load_rnn_data_by_batch(batch_ids, vocab_dict, x_path='processed/text'):
    X = np.zeros((len(batch_ids), MAX_NUM_WORD), dtype=np.int32)
    for i, ID in enumerate(batch_ids):
        with open(x_path+'/'+ID) as r_f:
            text = r_f.read() 
            doc = [vocab_dict[w] for w in text.split() if w in vocab_dict]
            X[i, :len(doc)] = doc
    return X