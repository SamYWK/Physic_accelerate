# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:34:46 2018

@author: SamKao
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

df = pd.read_csv('minitree_4b_2_26.txt', sep = ' ', header=0)

Jet_pt = df['Jet_pt'].values
Jet_genjetPt = df['Jet_genjetPt'].values
y = (Jet_pt/Jet_genjetPt).reshape(-1, 1)
X = df.drop(['Jet_pt'], axis = 1)
X = X.drop(['Jet_genjetPt'], axis = 1).values
print(X.shape)
print(y.shape)
learning_rate = 0.00001
batch_size = 256
g_1 = tf.Graph()
with g_1.as_default():
    X_placeholder = tf.placeholder(tf.float32, [None, 17])
    y_placeholder = tf.placeholder(tf.float32, [None, 1])
    
    a1 = tf.layers.dense(X_placeholder, 256, tf.nn.relu, name = 'layer_1')
    a2 = tf.layers.dense(a1, 256, tf.nn.relu, name = 'layer_2')
    a3 = tf.layers.dense(a2, 256, tf.nn.relu, name = 'layer_3')
    a4 = tf.layers.dense(a3, 256, tf.nn.relu, name = 'layer_4')
    a5 = tf.layers.dense(a4, 256, tf.nn.relu, name = 'layer_5')
    z6 = tf.layers.dense(a5, 1, name = 'layer_6')
    with tf.device('/gpu:0'):
        loss = tf.losses.mean_squared_error(labels = y_placeholder, predictions = z6)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #initializer
    init = tf.global_variables_initializer()
    
    with tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)) as sess:
        sess.run(init)
        for batch in range(1000):
            batch_xs = X[(batch*batch_size) : (batch+1)*batch_size]
            batch_ys = y[(batch*batch_size) : (batch+1)*batch_size]
            sess.run(train_step, feed_dict = {X_placeholder:batch_xs, y_placeholder:batch_ys})
            if batch % 10 == 0:
                print(sess.run(loss, feed_dict = {X_placeholder:batch_xs, y_placeholder:batch_ys}))