# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:26:37 2018

@author: SamKao
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
#import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sklearn.metrics
import time

df = pd.read_csv('minitree_4b_2_26.csv', header = 0)
#y = pd.concat((df['Jet_genjetPt'], df['Target']), axis = 1)
y = df['Jet_pt'].values.reshape(-1, 1)
X = df.drop(['Target', 'Jet_pt', 'Jet_genjetPt'], axis = 1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

learning_rate = 0.00001
batch_size = 100
epochs = 200
g_1 = tf.Graph()
n, d = X_train.shape

with g_1.as_default():
    with tf.device('/device:GPU:0'):
        X_placeholder = tf.placeholder(tf.float32, [None, 17])
        y_placeholder = tf.placeholder(tf.float32, [None, 1])
    
        a1 = tf.layers.dense(X_placeholder, 40, tf.nn.relu, name = 'layer_1')
        a2 = tf.layers.dense(a1, 40, tf.nn.relu, name = 'layer_2')
        a3 = tf.layers.dense(a2, 40, tf.nn.relu, name = 'layer_3')
        a4 = tf.layers.dense(a3, 40, tf.nn.relu, name = 'layer_4')
        a5 = tf.layers.dense(a4, 40, tf.nn.relu, name = 'layer_5')
        z6 = tf.layers.dense(a5, 1, name = 'layer_6')
    
        loss = tf.losses.mean_squared_error(labels = y_placeholder, predictions = z6)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #initializer
    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    
    #saver
    saver = tf.train.Saver()
    with tf.Session(config = config) as sess:
        sess.run(init)
        #saver.restore(sess, "./saver/model.ckpt")
        
        for epoch in range(epochs):
            for batch in range(int (n / batch_size)):
                batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                sess.run(train_step, feed_dict = {X_placeholder:batch_xs, y_placeholder:batch_ys})
                if batch % 1000:
                    print(sess.run(loss, feed_dict = {X_placeholder:batch_xs, y_placeholder:batch_ys}))
        #saver.save(sess, "./saver/model.ckpt")
        
        #pred = sess.run(z6, feed_dict = {X_placeholder:X_test})
        '''
        Jet_pt_hat = pred * Jet_pt
        error = abs(Jet_pt_hat - predict_target)
        count = 0
        print(len(error))
        for i in range(len(error)):
            if error[i]>0.8 and error[i]<1.6:
                count+=1
        print(count)
        
        print('\nR2_score: ', sklearn.metrics.r2_score(y_true = y_test, y_pred = pred))'''