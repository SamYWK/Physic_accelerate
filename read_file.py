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

reader = pd.read_table('minitree_4b_2_26.txt', sep = ' ')
df = reader
#df = pd.DataFrame(reader.get_chunk(500000))

#print(df.iloc[0])

Jet_genjetPt = df['Jet_genjetPt'].values
Jet_pt = df['Jet_pt'].values
y = (Jet_pt/Jet_genjetPt).reshape(-1, 1)
X = df.drop(['Jet_genjetPt', 'Jet_pt'], axis = 1).values.astype(np.float64)
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)
X = np.append(X, X[0].reshape(1, 17))
print(X.shape)
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

n, d = X_train.shape
batch_size = 200
epochs = 10
learning_rate = 0.00001
g_1 = tf.Graph()
prediction = np.array([])

with g_1.as_default():
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
        
        for epoch in range(epochs):
            for batch in range(int (n / batch_size)):
                batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                
                sess.run([train_step], feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys})
                if batch % 100 == 0:
                    print(sess.run([loss], feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys}))
        
        prediction = np.append(prediction,sess.run(z6, feed_dict = {X_placeholder : X_test}))
print('Prediction :', prediction[0])
print('Truth :', y_test[0])