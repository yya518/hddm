#!/usr/bin/python
#coding:utf-8
from __future__ import division
from topic_model import VariationalTopicModel
import tensorflow as tf
import time
import pickle
import numpy as np
import random
import pandas as pd
import os
from utilities import *
from sklearn.utils import shuffle

# hyperparameters
tf.flags.DEFINE_integer("num_topics", 50, "number of topics")
tf.flags.DEFINE_integer("vector_size", 100, "Dimensionality of topic vector")
tf.flags.DEFINE_integer("hidden_size", 64, "Dimensionality of hidden layer")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
tf.flags.DEFINE_integer("vocab_size", 2000, "vocabulary size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "keep probability for dropout")

tf.flags.DEFINE_integer("pretrain_epoch", 0, "pretrain")
tf.flags.DEFINE_string("out_dir", '', "output directory")
tf.flags.DEFINE_string("ckpt_name", '', "checkpoint to be restored")

FLAGS = tf.flags.FLAGS

# load data ids and labels
train_ids = np.load('processed/train_ids_stemmed.npy')
test_ids = np.load('processed/test_ids_stemmed.npy')
text_path = 'processed/stemmed_text'
# text_path = 'processed/text'
vocab_path = 'processed/stemmed_vocab_10k.txt'

# python train_topic_model.py --vocab_size 2000 --vector_size 100 --hidden_size 64 --dropout_keep_prob 0.8 --num_topics 100
# python train_topic_model.py --vocab_size 6000 --vector_size 100 --hidden_size 256 --dropout_keep_prob 0.8
# python train_topic_model.py --num_topics 100

def train_step(sess, model, x_batch):
    feed_dict = {
        model.x: x_batch,
        model.is_training: True
    }
    _, step, summaries, likelihood_cost, kl_cost, var_cost, perpl = sess.run([model.train_op, model.global_step, model.train_summary_op, model.generative_loss, model.inference_loss, model.variational_loss, model.perp], feed_dict)

    print("train: step {}, likelihood loss {:.2f}, kl_div loss {:.2f}, variantional loss {:.2f}, perplexity {:.2f}".format(
            step, likelihood_cost, kl_cost, var_cost, perpl))
    model.train_summary_writer.add_summary(summaries, step)

def dev_step(sess, model, ids, reduced_vocab, perp_record, epoch, mode="batch"):
    if mode == "whole":
        feed_dict = {
            model.x: load_bow_by_batch(ids, reduced_vocab, text_path),
            model.is_training:False
        }
        summaries, cost, perp = sess.run([model.dev_summary_op, model.generative_loss, model.perp], feed_dict)
    elif mode=="batch":
        perp_list= []
        cost_list = []
        num_batch = len(ids)//FLAGS.batch_size
        print('number of test data: ', len(ids))
        print('number of test steps: ', num_batch)
        for i in range(num_batch):
            batch_ids = ids[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
            feed_dict = {
                model.x: load_bow_by_batch(batch_ids, reduced_vocab, text_path),
                model.is_training:False
            }
            likelihood_cost, kl_cost, perp = sess.run([model.generative_loss, model.inference_loss, model.perp], feed_dict)
            print("test: step {}, likelihood loss {:.2f}, kl_div loss {:.2f}, perplexity {:.2f}".format(i, likelihood_cost, kl_cost, perp))
            perp_list.append(perp)
            cost_list.append(likelihood_cost)
        perp = np.mean(perp_list)
        cost = np.mean(cost_list)
        summaries = sess.run(model.dev_summary_op, {model.dev_perp: perp, model.dev_likelihood: cost})
    model.dev_summary_writer.add_summary(summaries, epoch)
    print("** dev **: epoch {}, likelihood loss {:.2f}, perplexity {:.2f} ".format(epoch, cost, perp))
    if perp < perp_record:
        perp_record = perp
        model.best_saver.save(sess, model.checkpoint_dir + '/best-model-prep={:.2f}-epoch{}.ckpt'.format(perp_record, epoch))
        print("new best dev perplexity: ", perp_record)
    else:
        print("the best dev perplexity: ", perp_record)
    model.current_saver.save(sess, model.checkpoint_dir + '/cur-model-prep={:.2f}-epoch{}.ckpt'.format(perp, epoch))    
    if epoch % 10 == 0:
        model.recent_saver.save(sess, model.checkpoint_dir + '/model-prep={:.2f}-epoch{}.ckpt'.format(perp, epoch))
    return perp_record

def main(conti=False):
    # load vocab
    reduced_vocab = [] 
    # vocab_path = "processed/topic_model_vocab_10000.txt"
    with open(vocab_path) as r_f:
        for i, l in enumerate(r_f):
            reduced_vocab.append(l.strip())
            if i == FLAGS.vocab_size-1:
                break

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True # allocate only as much GPU memory based on runtime allocations
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(config=config) as sess:
        with tf.variable_scope('variational_topic_model'):
            model = VariationalTopicModel(vocab_size=FLAGS.vocab_size,
                            latent_dim = FLAGS.hidden_size,
                            num_topic = FLAGS.num_topics,
                            vector_size=FLAGS.vector_size,
                            dropout_keep_proba=FLAGS.dropout_keep_prob
                            )
        if conti:
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_NTM_stemmed/{}".format(FLAGS.num_topics), ''))            
        else:
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_NTM_stemmed/{}".format(FLAGS.num_topics), timestamp)) 
        
        model.train_settings(out_dir, FLAGS.learning_rate, sess)
        
        if conti:
            model.recent_saver.restore(sess, model.checkpoint_dir + '')
            print("Model restored.")

        perp_record = np.inf

        num_train = len(train_ids)
        num_step_per_epoch = num_train//FLAGS.batch_size
        print('number of training data: ', num_train)
        print('number of steps per epcoh: ', num_step_per_epoch)
        for epoch in range(1, FLAGS.num_epochs+1):
            ID_train = shuffle(train_ids)
            ID_test = shuffle(test_ids)
            print('current epoch {}/{}'.format(epoch, FLAGS.num_epochs))
            for i in range(num_step_per_epoch):
                batch_ids = ID_train[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
                x_batch = load_bow_by_batch(batch_ids, reduced_vocab, text_path)
                train_step(sess, model, x_batch)
                # if i == int(num_step_per_epoch*0.25):
                #     perp_record = dev_step(sess, model, ID_test, reduced_vocab, perp_record, epoch-0.75, "batch")
                # if i == int(num_step_per_epoch*0.5):
                #     perp_record = dev_step(sess, model, ID_test, reduced_vocab, perp_record, epoch-0.5, "batch")
                # elif i == int(num_step_per_epoch*0.75):
                #     perp_record = dev_step(sess, model, ID_test, reduced_vocab, perp_record, epoch-0.25, "batch")
            perp_record = dev_step(sess, model, ID_test, reduced_vocab, perp_record, epoch, "batch")

if __name__ == '__main__':
    main(conti=False)