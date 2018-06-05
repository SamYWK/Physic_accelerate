# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:26:37 2018

@author: SamKao
"""

import pandas as pd
import tensorflow as tf
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

learning_rate = 0.001
batch_size = 100
epochs = 100

def add_layrer(inputs, in_dim, out_dim, activation = None, name = 'layer'):
    W = tf.Variable(tf.truncated_normal([in_dim, out_dim]))
    b = tf.Variable(tf.zeros([out_dim]))
    if activation == None:
        output = tf.matmul(inputs, W) + b
    else:
        output = activation(tf.matmul(inputs, W) + b)
    return output

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
        
            a1 = add_layrer(X_placeholder, 18, 100, tf.nn.tanh, name = 'layer_1')
            a2 = add_layrer(a1, 100, 100, tf.nn.tanh, name = 'layer_2')
            a3 = add_layrer(a2, 100, 100, tf.nn.tanh, name = 'layer_3')
            a4 = add_layrer(a3, 100, 100, tf.nn.tanh, name = 'layer_4')
            a5 = add_layrer(a4, 100, 75, tf.nn.tanh, name = 'layer_5')
            a6 = add_layrer(a5, 75, 15, tf.nn.tanh, name = 'layer_6')
            a7 = add_layrer(a6, 15, 10, tf.nn.tanh, name = 'layer_7')
            a8 = add_layrer(a7, 10, 1, name = 'layer_8')
        
            loss = tf.losses.mean_squared_error(labels = y_placeholder, predictions = a8)
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