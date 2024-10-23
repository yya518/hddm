import tensorflow as tf
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/util')
from layers import Layers

class TopicModel(object):

    def __init__(self, d, use_kl=0, all_likelihood = True,
        latent_sizes = [64, 32, 16, 8], layer_sizes = [512, 256, 128, 128], embedding_sizes = [100, 50, 20, 20]):

        """ model architecture """
        self.MLP_SIZES = layer_sizes
        self.Z_SIZES   = latent_sizes
        self.L = len(self.MLP_SIZES) # number of latent variables 
        self.embedding_size = embedding_sizes #[100, 100, 100, 100, 100]
        self.dropout_keep_proba = 0.5

        """ flags for regularizers """
        self.use_kl = use_kl
        self.all_likelihood = all_likelihood

        """ data and external toolkits """
        self.d  = d  # dataset manager
        self.ls = Layers()

        """ add palceholders """
        self.x = tf.placeholder(tf.float32, [self.d.batch_size, self.d.vocab_size], name="input_x")
        self.lr = tf.placeholder(tf.float32, shape=[], name="learning_rate") # learning rate
        self.lambda_z_wu = tf.placeholder(tf.float32, shape=[], name="lambda_z_wu") # warm up rate
        self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')

        """ cache for mu and sigma for z"""
        self.e_mus, self.e_logsigmas = [0]*self.L, [0]*self.L  # q(z_i+1 | z_i), bottom-up inference as Eq.7-9
        self.p_mus, self.p_logsigmas = [0]*self.L, [0]*self.L  # p(z_i | z_i+1), top-down prior as Eq.1-3
        self.d_mus, self.d_logsigmas = [0]*self.L, [0]*self.L  # q(z_i | .), bidirectional inference as Eq.17-19
        self.z = [0]*self.L

        """ cache for theta """
        self.topic_dist = [0]*self.L

        """ prior of theta """
        self.W = [0]*self.L
        self.topic_embed = [0]*self.L
        self.beta = [self.build_beta(l) for l in range(self.L)] # supertopic-subtopic distribution
        self.likelihood = [0]*self.L 

        self.build_graph()

    def build_beta(self, l):
        self.W[l] = self.ls.W([self.d.vocab_size, self.embedding_size[l]], W_name="word_embed" + str(l))
        W_dropped = tf.contrib.layers.dropout(self.W[l], 
                    keep_prob=self.dropout_keep_proba, is_training=self.is_train)
        self.topic_embed[l] = self.ls.W([self.Z_SIZES[l], self.embedding_size[l]], W_name="topic_embed" + str(l))
        # beta: topic-word distribution. shape: [num_topic, vocab_size]
        return tf.nn.softmax(tf.matmul(self.topic_embed[l], W_dropped, transpose_b=True))

    def encoder(self):

        h = self.x
        for l in range(self.L):
            scope = 'Encode_L' + str(l)
            h = self.ls.dense(scope, h, self.MLP_SIZES[l])
            h = self.ls.bn(scope, h, self.is_train, name=scope)
            h = tf.nn.elu(h) # Computes exponential linear: exp(features) - 1 if < 0, features otherwise.

            """ prepare for bidirectional inference """
            _, self.e_mus[l], self.e_logsigmas[l] = self.ls.vae_sampler(
                scope, h, self.Z_SIZES[l], tf.nn.softplus # Computes softplus: log(exp(features) + 1)
            ) # Eq.13-15

    def decoder(self):

        for l in range(self.L-1, -1, -1):
            scope = 'Decoder_L' + str(l)

            if l == self.L-1:
                """ At the highest latent layer, mu & sigma are identical to those outputed from encoder.
                    And making actual z is not necessary for the highest layer."""
                self.d_mus[l], self.d_logsigmas[l] = self.e_mus[l], self.e_logsigmas[l]
                self.z[l] = self.ls.sampler(self.d_mus[l], tf.exp(self.d_logsigmas[l]))

            else:
                """ prior is developed from z of the above layer """
                _, self.p_mus[l], self.p_logsigmas[l] = self.ls.vae_sampler(
                                                        scope, self.topic_dist[l+1], self.Z_SIZES[l], tf.nn.softplus
                                                    ) # Eq.13-15

                self.z[l], self.d_mus[l], self.d_logsigmas[l] = self.ls.precision_weighted_sampler(
                        scope,
                        (self.e_mus[l], tf.exp(self.e_logsigmas[l])),
                        (self.p_mus[l], tf.exp(self.p_logsigmas[l]))
                    )  # Eq.17-19
            
            self.topic_dist[l] = self.ls.dense('topic' + str(l), self.z[l], self.Z_SIZES[l], tf.nn.softmax)
            p_x = tf.matmul(self.topic_dist[l], self.beta[l])
            self.likelihood[l] = -tf.reduce_sum(self.x * tf.log(p_x + 1e-10), 1)

    def get_z_kl(self, l):
        d_sigma2 = tf.exp(2 * self.d_logsigmas[l])
    
        kl = 0.5 * tf.reduce_sum(tf.square(self.d_mus[l]) + d_sigma2 - 2 * self.d_logsigmas[l] - 1.0, 1)

        return kl

    def build_graph(self):

        o = dict()  # output
        self.encoder()
        self.decoder()
        loss = 0.0

        """ p(x|z) Reconstruction Loss """
        o['Lr'] = [0]*self.L
        for l in range(self.L):
            o['Lr'][l] = tf.reduce_mean(self.likelihood[l])
            if self.all_likelihood:
                loss += o['Lr'][l]
            else:
                loss = o['Lr'][0]

        # calculate the perplexity of the topic model using likelihood
        # prep: scalar
        o['perp_all'] = [0]*self.L
        o['perp_perdoc'] = [0]*self.L
        for l in range(self.L):
            o['perp_all'][l] = tf.exp(tf.reduce_sum(self.likelihood[l]) / tf.reduce_sum(self.x))
            o['perp_perdoc'][l] = tf.reduce_mean(tf.exp(self.likelihood[l] / tf.reduce_sum(self.x, 1)))

        """ VAE KL-Divergence Loss """
        o['kl'] = [0]*self.L
        for l in range(self.L):
            o['kl'][l] = tf.reduce_mean(self.get_z_kl(l))

        if self.use_kl == 1:
            loss += self.lambda_z_wu * o['kl'][-1]
        elif self.use_kl == 2:
            for kl in o['kl']:
                loss += self.lambda_z_wu * kl
        
        # tf.cond(self.use_kl, lambda: o['Lr'] + self.lambda_z_wu * kl, lambda: o['Lr'])

        """ set losses """
        o['loss'] = loss
        self.out = o

        """ set optimizer """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
        grads = optimizer.compute_gradients(loss)
        for i,(g,v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g,5),v) # clip gradients
            else:
                print('g is None:', v)
                v = tf.Print(v, [v], "v = ", summarize=10000)
        self.op = optimizer.apply_gradients(grads) # return train_op

