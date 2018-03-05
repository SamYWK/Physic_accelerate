# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 10:46:06 2018

@author: pig84
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

CSV_TYPES = [[0.0]]*19
g_1 = tf.Graph()
learning_rate = 0.00001
filenames = ['minitree_4b_2_26.txt']

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES, field_delim = ' ')
    # Pack the result into a dictionary
    #data = dict(zip(CSV_COLUMN_NAMES, fields))
    # Separate the label from the features

    label = fields[30]
    label = tf.reshape(label, [1])
    features = tf.stack(fields[0:30])
    return features, label

with g_1.as_default():
    dataset = tf.data.TextLineDataset(filenames).skip(1)
    dataset = dataset.map(_parse_line)
    dataset = dataset.shuffle(1000).repeat().batch(100)
    
    X_placeholder = tf.placeholder(tf.float64, [None, 17])
    y_placeholder = tf.placeholder(tf.float64, [None, 1])
    
    a1 = tf.layers.dense(X_placeholder, 35, tf.nn.relu, name = 'layer_1')
    a2 = tf.layers.dense(a1, 35, tf.nn.relu, name = 'layer_2')
    a3 = tf.layers.dense(a2, 25, tf.nn.relu, name = 'layer_3')
    a4 = tf.layers.dense(a3, 25, tf.nn.relu, name = 'layer_4')
    a5 = tf.layers.dense(a4, 15, tf.nn.relu, name = 'layer_5')
    z6 = tf.layers.dense(a5, 1, name = 'layer_6')
    
    loss = tf.losses.mean_squared_error(labels = y_placeholder, predictions = z6)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    #prediction
    #correct_prediction = tf.equal(tf.argmax(x1,1), tf.argmax(y_placeholder,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #initializer
    init = tf.global_variables_initializer()
    
    with tf.Session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
        sess.run(init)
        #writer
        #writer = tf.summary.FileWriter('logs/', sess.graph)
        
        