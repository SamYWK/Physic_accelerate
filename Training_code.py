# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:26:37 2018

@author: SamKao
"""

import pandas as pd
import tensorflow as tf
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

learning_rate = 0.00001
batch_size = 100
epochs = 150


def main():
    #data preprocessing
    df = pd.read_table('minitree_4b_2_26.txt', header = 0, sep = ' ')
    df['Target'] = (df['Jet_genjetPt']/df['Jet_pt']).values
    y = df['Target'].values.reshape(-1, 1)
    X = df.drop(['Target', 'Jet_genjetPt'], axis = 1).values
    X_train = X 
    y_train = y
    
    #define graph
    g_1 = tf.Graph()
    n, d = X_train.shape
    
    with g_1.as_default():
        with tf.device('/device:GPU:0'):
            X_placeholder = tf.placeholder(tf.float32, [None, 18])
            y_placeholder = tf.placeholder(tf.float32, [None, 1])
        
            a1 = tf.layers.dense(X_placeholder, 15, tf.nn.relu, name = 'layer_1')
            a2 = tf.layers.dense(a1, 15, tf.nn.relu, name = 'layer_2')
            a3 = tf.layers.dense(a2, 15, tf.nn.relu, name = 'layer_3')
            a4 = tf.layers.dense(a3, 15, tf.nn.relu, name = 'layer_4')
            a5 = tf.layers.dense(a4, 15, tf.nn.relu, name = 'layer_5')
            a6 = tf.layers.dense(a5, 15, tf.nn.relu, name = 'layer_6')
            a7 = tf.layers.dense(a6, 15, tf.nn.relu, name = 'layer_7')
            a8 = tf.layers.dense(a7, 15, tf.nn.relu, name = 'layer_8')
            a9 = tf.layers.dense(a8, 15, tf.nn.relu, name = 'layer_9')
            a10 = tf.layers.dense(a9, 13, tf.nn.relu, name = 'layer_10')
            a11 = tf.layers.dense(a10, 13, tf.nn.relu, name = 'layer_11')
            a12 = tf.layers.dense(a11, 10, tf.nn.relu, name = 'layer_12')
            a13 = tf.layers.dense(a12, 10, name = 'layer_13')
            z14 = tf.layers.dense(a13, 1, name = 'layer_14')
        
            loss = tf.losses.mean_squared_error(labels = y_placeholder, predictions = z14)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            
        #initialize all variables
        init = tf.global_variables_initializer()
    
        config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        
        #saver
        saver = tf.train.Saver()
        with tf.Session(config = config) as sess:
            sess.run(init)
            #saver.restore(sess, "./saver/model.ckpt")
            
            #trianing part
            for epoch in range(epochs):
                for batch in range(int (n / batch_size)):
                    batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                    batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                    sess.run(train_step, feed_dict = {X_placeholder:batch_xs, y_placeholder:batch_ys})
                    if batch % 50000:
                        #every 50000 batch print loss
                        print(sess.run(loss, feed_dict = {X_placeholder:batch_xs, y_placeholder:batch_ys}))
            
            #save parameters
            saver.save(sess, "./saver/model.ckpt")

if __name__ == "__main__":
    main()