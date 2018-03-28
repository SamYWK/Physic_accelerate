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
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sklearn.metrics

df = pd.read_csv('minitree_4b_2_26.csv', header = 0)
#y = pd.concat((df['Jet_genjetPt'], df['Target']), axis = 1)
y = df['Target'].values.reshape(-1, 1)
X = df.drop(['Target', 'Jet_genjetPt', 'Jet_pt'], axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

learning_rate = 0.000001
batch_size = 100
epochs = 5
g_1 = tf.Graph()
n, d = X_train.shape

def add_layer(inputs, in_dim, out_dim, activation = None, name = 'layer'):
    with tf.name_scope(name):
        with tf.name_scope('Weights'):
            W = tf.Variable(tf.truncated_normal([in_dim, out_dim]))
        with tf.name_scope('bias'):
            b = tf.Variable(tf.zeros([out_dim]))
        with tf.name_scope('Wx_plus_b'):
            if activation == None:
                output = tf.matmul(inputs, W) + b
            else:
                output = activation(tf.matmul(inputs, W) + b)
        return output

with g_1.as_default():
    with tf.device('/device:GPU:1'):
        X_placeholder = tf.placeholder(tf.float32, [None, 17])
        y_placeholder = tf.placeholder(tf.float32, [None, 1])
    
        a1 = add_layer(X_placeholder, 17, 15, tf.nn.sigmoid, name = 'layer_1')
        a2 = add_layer(a1, 15, 15, tf.nn.sigmoid, name = 'layer_2')
        a3 = add_layer(a2, 15, 15, tf.nn.sigmoid, name = 'layer_3')
        a4 = add_layer(a3, 15, 15, tf.nn.sigmoid, name = 'layer_4')
        a5 = add_layer(a4, 15, 10, tf.nn.sigmoid, name = 'layer_5')
        a6 = add_layer(a5, 10, 10, tf.nn.sigmoid, name = 'layer_6')
        a7 = add_layer(a6, 10, 10, tf.nn.sigmoid, name = 'layer_7')
        a8 = add_layer(a7, 10, 5, name = 'layer_8')
        z9 = add_layer(a8, 5, 1, name = 'layer_9')
    
        loss = tf.losses.mean_squared_error(labels = y_placeholder, predictions = z9)
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
                if batch % 100000:
                    print(sess.run(loss, feed_dict = {X_placeholder:batch_xs, y_placeholder:batch_ys}))
        saver.save(sess, "./saver/model.ckpt")
        
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