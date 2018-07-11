# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:26:37 2018

@author: SamKao
"""

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

learning_rate = 1e-4
batch_size = 6400
epochs = 200

def main():
    #data preprocessing
    df = pd.read_table('minitree_4b_leading_2_26.txt', header = 0, sep = ' ')
#    df = pd.read_csv('./minitree_4b_2_26_modified.csv', header = 0)
    df['Target'] = (df['Jet_genjetPt']/df['Jet_pt']).values
    y = df['Target'].values.reshape(-1, 1)
    X = df.drop(['Target', 'Jet_genjetPt'], axis = 1).values
    #normalize X
#    scaler = MinMaxScaler()
#    X = scaler.fit_transform(X)
    np.random.shuffle(X)
    X_train = X
    np.random.shuffle(y)
    y_train = y
    
    #define graph
    g_1 = tf.Graph()
    n, d = X_train.shape
    
    with g_1.as_default():
        with tf.device('/device:GPU:0'):
            X_placeholder = tf.placeholder(tf.float32, [None, 18])
            y_placeholder = tf.placeholder(tf.float32, [None, 1])
            
            regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-2)
        
            a1 = tf.layers.dense(X_placeholder, 18, activation = tf.nn.relu, name = 'layer_1', kernel_regularizer = regularizer)
            a2 = tf.layers.dense(a1, 50, activation = tf.nn.relu, name = 'layer_2', kernel_regularizer = regularizer)
            a3 = tf.layers.dense(a2, 50, activation = tf.nn.relu, name = 'layer_3', kernel_regularizer = regularizer)
            a4 = tf.layers.dense(a3, 1, activation = None, name = 'layer_4', kernel_regularizer = regularizer)
            
            loss = tf.losses.huber_loss(predictions = a4, labels = y_placeholder)# + tf.losses.get_regularization_loss()
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            
        #initialize all variables
        init = tf.global_variables_initializer()
    
        config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        
        #saver
        saver = tf.train.Saver()
        with tf.Session(config = config) as sess:
            sess.run(init)
#            saver.restore(sess, "./saver_3_layers_10_epochs_test/model.ckpt")
            
            #trianing part
            for epoch in range(epochs):
                for batch in range(int (n / batch_size)):
                    batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                    batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                    sess.run(train_step, feed_dict = {X_placeholder:batch_xs, y_placeholder:batch_ys})
                    if batch % 50000 == 0:
                        print('Epoch :', epoch, 
                              'Loss :', sess.run(loss, feed_dict = {X_placeholder:batch_xs, y_placeholder:batch_ys}))
            #save parameters
            saver.save(sess, "./saver/model.ckpt")
if __name__ == "__main__":
    main()