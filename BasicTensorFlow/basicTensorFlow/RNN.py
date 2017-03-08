'''
Created on Mar 1, 2017

@author: mozhu
'''

'''
This is for RNN with sequence data training
Here is config:
x = [1,5]; s=[1,3]; w=[3,3], u=[5,3], v=[3,5], y=[1,5]
st = tanh(xu + st-1*w)
loss = entropy loss
'''

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [6,5])
y = tf.placeholder(tf.float32, [6,5])
S = tf.placeholder(tf.float32, [1,3])
W = tf.Variable(tf.random_normal([3, 3], stddev = 0.35, mean=2))
U = tf.Variable(tf.random_normal([5, 3], stddev = 0.35, mean=2))
V = tf.Variable(tf.random_normal([3, 5], stddev = 0.35, mean=2))

current_state = tf.nn.tanh(tf.matmul(x, U) + tf.matmul(S, W))
prediction_model = tf.nn.softmax(tf.matmul(current_state, V))
#cost = 