# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 22:27:46 2018

@author: pig84
"""

import tensorflow as tf
import pandas as pd

def main():
    #Read testing data
    #without Jet_genjetPt, which means column = 18
    #X_test = pd.read_table('./xxxx.txt', header = 0, sep = ' ')
    
    #with Jet_genjetPt, which means column = 19
    X_test = pd.read_table('./minitree_4b_2_26.txt', header = 0, sep = ' ').drop(['Jet_genjetPt'], axis = 1)
    
    #defining graph
    g_1 = tf.Graph()
    
    with g_1.as_default():
        with tf.device('/device:GPU:0'):
            X_placeholder = tf.placeholder(tf.float32, [None, 18])
        
            a1 = tf.layers.dense(X_placeholder, 15, tf.nn.sigmoid, name = 'layer_1')
            a2 = tf.layers.dense(a1, 15, tf.nn.sigmoid, name = 'layer_2')
            a3 = tf.layers.dense(a2, 15, tf.nn.sigmoid, name = 'layer_3')
            a4 = tf.layers.dense(a3, 15, tf.nn.sigmoid, name = 'layer_4')
            a5 = tf.layers.dense(a4, 15, tf.nn.sigmoid, name = 'layer_5')
            a6 = tf.layers.dense(a5, 15, tf.nn.sigmoid, name = 'layer_6')
            a7 = tf.layers.dense(a6, 10, tf.nn.sigmoid, name = 'layer_7')
            a8 = tf.layers.dense(a7, 1, name = 'layer_8')
        
        config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        
        #saver
        saver = tf.train.Saver()
        with tf.Session(config = config) as sess:
            #restore parameters form saver
            saver.restore(sess, "./saver/model.ckpt")
            #make prediction
            prediction = sess.run(a8, feed_dict = {X_placeholder:X_test})
            
            #write prediction in text file
            text_file = open('prediction.txt', 'w')
            text_file.write('Jet_genjetPt\n')
            for i in range(len(prediction)):
                text_file.write(str(prediction[i]) + '\n')
            text_file.close()
            
if __name__ == '__main__':
    main()