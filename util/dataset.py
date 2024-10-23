import tensorflow as tf
import numpy as np
import sys, os, time
from gensim.corpora.dictionary import Dictionary
import pickle

class Dataset(object):

    def __init__(self, dataset, batch_size=None, vocab_size=2000):
        self.dataset = dataset
        self.batch_size = batch_size 
        self.vocab_size = vocab_size
        if self.dataset == "20news":
            self.load20news()
        elif self.dataset == "mr":
            self.loadMRD()
        elif self.dataset == "wiki_company":
            self.loadwiki_company()

    def load20news(self):
        self.vocab = Dictionary.load('./20news/vocab2000.pkl')
        self.corpus = pickle.load(open("./20news/corpus.pkl", 'rb'))
        self.texts = pickle.load(open("./20news/texts.pkl", 'rb'))

        BOW = np.zeros([len(self.corpus), self.vocab_size])
        for i, bow in enumerate(self.corpus):
            for word_id, count in bow:
                BOW[i][word_id] = count
        self.num_train_batch = int(len(self.corpus)*0.9)//self.batch_size
        self.train_x_bow = BOW[:self.num_train_batch*self.batch_size]
        self.num_test_batch = len(self.corpus)//self.batch_size - self.num_train_batch
        self.test_x_bow = BOW[self.num_train_batch*self.batch_size:]
    
    
    def loadMRD(self):
        self.vocab = Dictionary.load('./movie_review/vocab2000.pkl')
        self.corpus = pickle.load(open("./movie_review/corpus.pkl", 'rb'))
        self.texts = pickle.load(open("./movie_review/texts.pkl", 'rb'))

        BOW = np.zeros([len(self.corpus), self.vocab_size])
        for i, bow in enumerate(self.corpus):
            for word_id, count in bow:
                BOW[i][word_id] = count
        self.num_train_batch = int(len(self.corpus)*0.7)//self.batch_size
        self.train_x_bow = BOW[:self.num_train_batch*self.batch_size]
        self.num_test_batch = len(self.corpus)//self.batch_size - self.num_train_batch
        self.test_x_bow = BOW[self.num_train_batch*self.batch_size:]

        
    def loadwiki_company(self):        
        self.vocab = Dictionary.load('./wiki_company/vocab2000_new.pkl')
        self.corpus = pickle.load(open("./wiki_company/corpus_new.pkl", 'rb'))
        self.texts = pickle.load(open("./wiki_company/texts_new.pkl", 'rb'))

        BOW = np.zeros([len(self.corpus), self.vocab_size])
        for i, bow in enumerate(self.corpus):
            for word_id, count in bow:
                BOW[i][word_id] = count
        self.num_train_batch = int(len(self.corpus)*0.7)//self.batch_size
        self.train_x_bow = BOW[:self.num_train_batch*self.batch_size]
        self.num_test_batch = len(self.corpus)//self.batch_size - self.num_train_batch
        self.test_x_bow = BOW[self.num_train_batch*self.batch_size:]

            
    def load_any_text(self, gensim_vocab, gensim_corpus):
        self.vocab = gensim_vocab
        BOW = np.zeros([len(gensim_corpus), len(self.vocab)])
        for i, bow in enumerate(gensim_corpus):
            for word_id, count in bow:
                BOW[i][word_id] = count
        print(BOW)
        self.num_train_batch = int(len(gensim_corpus)*0.7)//self.batch_size
        self.train_x_bow = BOW[:self.num_train_batch*self.batch_size]
        self.num_test_batch = len(gensim_corpus)//self.batch_size - self.num_train_batch
        self.test_x_bow = BOW[self.num_train_batch*self.batch_size:]
