'''
train hierarchical neural topic on any text
python main.py --batch_size 64 --dir_logs "./out/wiki_company/" --num_epochs 300 --num_topics "128 64 32" --layer_sizes "512 256 128" --embedding_sizes "100 50 25" --dataset wiki_company
'''

import tensorflow as tf
import numpy as np
import sys, os, time, math
from model import TopicModel
from dataset import Dataset
from sklearn.utils import shuffle
import pandas as pd
import gensim.corpora as corpora
import pickle
import pandas as pd
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import strip_punctuation, strip_numeric, remove_stopwords, stem, strip_short
from gensim.parsing.preprocessing import preprocess_string


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

tf.flags.DEFINE_boolean("restore", False, "restore from the last check point")
tf.flags.DEFINE_string("dir_logs", "./out/", "out directory")
tf.flags.DEFINE_string("num_topics", "32 16 8", "number of topics (separated by space)")
tf.flags.DEFINE_string("layer_sizes", "512 256 128", "size of all latent layers (separated by space)")
tf.flags.DEFINE_string("embedding_sizes", "100 50 20", "size of embeddings in the topic-word distribution matrices (separated by space)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs")
tf.flags.DEFINE_integer("use_kl", 0, "none: 0; top one: 1; all: 2")
tf.flags.DEFINE_string("dataset", "wiki_company", "datasets in the experiment: 20news wiki_company movie_review mimic") 

FLAGS = tf.flags.FLAGS

num_topics = [int(e) for e in FLAGS.num_topics.strip().split()]
layer_sizes = [int(e) for e in FLAGS.layer_sizes.strip().split()]
embedding_sizes = [int(e) for e in FLAGS.embedding_sizes.strip().split()]

print('dataset: ', FLAGS.dataset)
print('num_topics: ', num_topics)
print('layer_sizes: ', layer_sizes)
print('embedding_sizes: ', embedding_sizes)

N_EPOCHS = FLAGS.num_epochs
BATCH_SIZE = FLAGS.batch_size
FILE_OF_CKPT  = os.path.join(FLAGS.dir_logs,"model.ckpt")

# learning rate decay
STARTER_LEARNING_RATE = FLAGS.learning_rate
DECAY_AFTER = 2
DECAY_INTERVAL = 5
DECAY_FACTOR = 0.97

# warming-up coefficient for KL-divergence term
Nt = 50 # warmig-up during the first 2Nt epochs
# _lambda_z_wu = np.concatenate((np.zeros(Nt), np.linspace(0, 1, Nt)))
_lambda_z_wu = np.linspace(0, 1, Nt)


d = Dataset(FLAGS.dataset, BATCH_SIZE)

print(d.train_x_bow)

print('training size: ', len(d.train_x_bow) )
print('validation size: ', len(d.test_x_bow) )




####################Training################################################
with tf.Graph().as_default() as g:

    ###########################################
    """        Build Model Graphs           """
    ###########################################
    with tf.variable_scope("topicmodel") as scope:
        m = TopicModel(d, FLAGS.use_kl, 
                       latent_sizes = num_topics, 
                       layer_sizes = layer_sizes, 
                       embedding_sizes = embedding_sizes)
        print('built the graph for training.')
        scope.reuse_variables()
    
    ###########################################
    """              Init                   """
    ###########################################
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    saver = tf.train.Saver()
    _lr, ratio = STARTER_LEARNING_RATE, 1.0
    Perp_all =[]
    Perp_perdoc = []

    if FLAGS.restore:
        print("... restore from the last check point.")
        saver.restore(sess, FILE_OF_CKPT)
    ###########################################
    """         Training Loop               """
    ###########################################
    print('... start training')
    tf.train.start_queue_runners(sess=sess)
    best_perp = math.inf
    for epoch in range(1, N_EPOCHS+1):
        X_train = shuffle(d.train_x_bow)
        X_test = shuffle(d.test_x_bow)  #this is validation
        # set coefficient of warm-up
        idx = -1 if Nt <= epoch else epoch
        time_start = time.time()
        perp_all = [0.0] * m.L
        perp_perdoc = [0.0] * m.L
        for i in range(d.num_train_batch):
        
            feed_dict = {m.x: X_train[i*d.batch_size: (i+1)*d.batch_size], 
                        m.lr:_lr, 
                        m.lambda_z_wu:_lambda_z_wu[idx],
                        m.is_train: True}

            """ do update """
            r, op, current_lr = sess.run([m.out, m.op, m.lr], feed_dict=feed_dict)
            for l in range(m.L):
                perp_all[l] += r['perp_all'][l]
                perp_perdoc[l] += r['perp_perdoc'][l]
        for l in range(m.L):
            perp_all[l] /= d.num_train_batch
            perp_perdoc[l] /= d.num_train_batch
        elapsed_time = time.time() - time_start
        print(" epoch:%2d, train loss: %s, likelihood: %s, KL: %s, perp_all: %s, perp_perdoc: %s, time:%.3f" %
                (epoch, r['loss'], r['Lr'], r['kl'], perp_all, perp_perdoc, elapsed_time))

        """ test """
        time_start = time.time()
        perp_all = [0.0] * m.L
        perp_perdoc = [0.0] * m.L
        for i in range(d.num_test_batch):
            feed_dict = {m.x: X_test[i*d.batch_size: (i+1)*d.batch_size], 
                        m.lambda_z_wu:_lambda_z_wu[idx],
                        m.is_train: False}
            r = sess.run([m.out], feed_dict=feed_dict)[0]
            for l in range(m.L):
                perp_all[l] += r['perp_all'][l]
                perp_perdoc[l] += r['perp_perdoc'][l]
        for l in range(m.L):
            perp_all[l] /= d.num_test_batch
            perp_perdoc[l] /= d.num_test_batch
        Perp_all.append(perp_all[0])
        Perp_perdoc.append(perp_perdoc[0])

        elapsed_time = time.time() - time_start
        print(" epoch:%2d, test loss: %s, likelihood: %s, KL: %s, perp_all: %s, perp_perdoc: %s, time:%.3f" %
                (epoch, r['loss'], r['Lr'], r['kl'], perp_all, perp_perdoc, elapsed_time))

        """ save """
        if epoch % 10 == 0:
            print("Model saved in file: %s" % saver.save(sess,FILE_OF_CKPT))
        if perp_all[0] < best_perp:
            best_perp = perp_all[0]
            print("Model saved in file: %s" % saver.save(sess, os.path.join(FLAGS.dir_logs,"best-model.ckpt")))
        
        print("best test perp: ", best_perp)

        """ learning rate decay"""
        if (epoch % DECAY_INTERVAL == 0) and (epoch > DECAY_AFTER):
            ratio *= DECAY_FACTOR
            _lr = STARTER_LEARNING_RATE * ratio
            print('lr decaying is scheduled. epoch:%d, lr:%f <= %f' % ( epoch, _lr, current_lr))

        np.save(os.path.join(FLAGS.dir_logs,"perp_all.npy"), Perp_all)
        np.save(os.path.join(FLAGS.dir_logs,"perp_perdoc.npy"), Perp_perdoc)

    sess.close()

    
    
    
##############EVALUATION#######################################################
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel

def get_topic_words(FILE_OF_CKPT):
    with tf.Graph().as_default() as g:

        ###########################################
        """        Build Model Graphs           """
        ###########################################
        with tf.variable_scope("topicmodel") as scope:
            m = TopicModel(d, latent_sizes = num_topics, 
                       layer_sizes = layer_sizes, 
                       embedding_sizes = embedding_sizes)
            print('built the graph for training.')
            scope.reuse_variables()

        ###########################################
        """              Init                   """
        ###########################################
        init_op = tf.global_variables_initializer()
        sess  = tf.Session()
        sess.run(init_op)
        saver = tf.train.Saver()

        print("... restore from the last check point.")
        saver.restore(sess, FILE_OF_CKPT)
        
        beta = sess.run([m.beta], feed_dict = {m.is_train: False})
        sess.close()
        
        return beta
    
def print_topic_to_list(topic_word_list, vocab, top_k=20):
    topic_id = 0
    topwords = []
    for l, topic_word in enumerate(np.flip(topic_word_list)): #for each level
        for i, t in enumerate(topic_word):
            term_idx = np.argsort(t)
            topKwords = []
            for j in np.flip(term_idx[-top_k:]):
                topKwords.append(vocab[j])
            print('topic', topic_id, ':', ' '.join(topKwords))
            topic_id+=1
            topwords.append(topKwords)
    return topwords


def print_topic_to_list_with_rerank(topic_word_list, vocab, top_k=20):
    topic_id = 0
    topwords = []
    for l, topic_word in enumerate(np.flip(topic_word_list)): #for each level
        word_sum_prob = np.sum(topic_word, axis=0)
        print(topic_word.shape)
        print(word_sum_prob.shape)
        
        normalized_word_prob = topic_word / word_sum_prob
        
        for i, t in enumerate(normalized_word_prob):
            term_idx = np.argsort(t)
            topKwords = []
            for j in np.flip(term_idx[-top_k:]):
                topKwords.append(vocab[j])
            print('topic', topic_id, ':', ' '.join(topKwords))
            topic_id+=1
            topwords.append(topKwords)
    return topwords

def proportion_uniqe_words(topics, topk=10):
    unique_words = set()
    for topic in topics:
        unique_words = unique_words.union(set(topic[:topk]))
    puw = len(unique_words) / (topk*len(topics))
    print('topic diversity: ', puw)
    

beta = get_topic_words(FLAGS.dir_logs+'best-model.ckpt')
topwords = print_topic_to_list_with_rerank(beta[0], d.vocab)            
proportion_uniqe_words(topwords, 10)

cm = CoherenceModel(topics=topwords, texts = d.texts, corpus = d.corpus, dictionary=d.vocab, coherence='c_v', window_size=5)
print('topic coherence: ', cm.get_coherence()  )