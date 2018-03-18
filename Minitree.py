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

df = pd.read_csv('minitree_4b_2_26.txt', sep = ' ', header=0)
target = (df['Jet_genjetPt']/df['Jet_pt']).values.reshape(-1, 1)
X = df.drop(['Jet_genjetPt'], axis = 1).values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
X_train = X[0:3200000]
y_train = target[0:3200000]
X_test = X[3200000:]
y_test = target[3200000:]

predict_target = df['Jet_genjetPt'].values[3200000:].reshape(-1, 1)
Jet_pt = df['Jet_pt'].values[3200000:].reshape(-1, 1)

learning_rate = 0.00001
batch_size = 100
epochs = 60
g_1 = tf.Graph()
n, d = X_train.shape

with g_1.as_default():
    
    X_placeholder = tf.placeholder(tf.float32, [None, 18])
    y_placeholder = tf.placeholder(tf.float32, [None, 1])

    a1 = tf.layers.dense(X_placeholder, 40, tf.nn.relu, name = 'layer_1')
    a2 = tf.layers.dense(a1, 50, tf.nn.relu, name = 'layer_2')
    a3 = tf.layers.dense(a2, 50, tf.nn.relu, name = 'layer_3')
    a4 = tf.layers.dense(a3, 50, tf.nn.relu, name = 'layer_4')
    a5 = tf.layers.dense(a4, 40, tf.nn.relu, name = 'layer_5')
    z6 = tf.layers.dense(a5, 1, name = 'layer_6', activation = None)

    loss = tf.losses.mean_squared_error(labels = y_placeholder, predictions = z6)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #initializer
    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement = False, log_device_placement=True)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    
    #saver
    saver = tf.train.Saver()
    with tf.Session(config = config) as sess:
        sess.run(init)
        #saver.restore(sess, "./saver/model.ckpt")
        
        for epoch in range(epochs):
            t_start = time.time()
            for batch in range(int (n / batch_size)):
                batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                sess.run(train_step, feed_dict = {X_placeholder:batch_xs, y_placeholder:batch_ys})
                
                sys.stdout.write("\rEpoch("+ str(epoch)+ "):" + str(sess.run(loss, feed_dict = {X_placeholder:batch_xs, y_placeholder:batch_ys})))
            t_stop = time.time()
            print("   It cost %f time"%(t_stop-t_start))
        saver.save(sess, "./saver2/model.ckpt")
        
        pred = sess.run(z6, feed_dict = {X_placeholder:X_test})
        '''
        Jet_pt_hat = pred * Jet_pt
        error = abs(Jet_pt_hat - predict_target)
        count = 0
        print(len(error))
        for i in range(len(error)):
            if error[i]>0.8 and error[i]<1.6:
                count+=1
        print(count)
        '''
        print('\nR2_score: ', sklearn.metrics.r2_score(y_true = y_test, y_pred = pred))