'''
Created on Feb 27, 2017

@author: mozhu
'''

'''
This is for logistic Regression
We have 
Here is logistic model: y = softmax(W*x)
Here is the lost: cross_entropy cost = -ylog(y~)

x = [1,5], X = [6,5]
W = [5,1]
y = [1,1], Y = [6,1]
class size == 2
'''

import tensorflow as tf
import numpy as np

W = tf.Variable(tf.random_normal([5, 2], stddev = 0.35, mean=2), name = "weight")

y_ = tf.placeholder(tf.float32, [6,2], name = "target")
x_ = tf.placeholder(tf.float32, [6,5], name = "input_data")

logistic_regression_model = tf.nn.softmax(tf.matmul(x_,W))
cost = -tf.reduce_mean(tf.reduce_sum(y_*tf.log(logistic_regression_model), 1))

'''
k_ = tf.placeholder(tf.float32, [2, 5], name = "testdata")
k_1 = tf.Variable(tf.random_normal([1, 5], stddev = 0.35, mean=2), name = "weight1")
test_softmax_model = tf.nn.softmax(k_)
'''
x_train = np.array([[1,2,3,4,5], [3,4,5,6,7], [9,7,4,2,1], [7,4,1,6,5], [1,5,3,6,7], [1,4,5,7,3]])
y_train = np.array([[1,0], [0,1],[0,1],[1,0],[0,1],[1,0]])

optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(cost)

#k_1 = np.array([1,2,3])
#k_2 = np.array([1,2,3])

#k_1p = tf.placeholder(tf.float32, [3])
#k_2p = tf.placeholder(tf.float32, [3])

#print(x_train[:,[0,1]])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
'''
res = sess.run(test_softmax_model,{k_:x_train[[1,2],:]})
print(res)
'''
#print(sess.run(k_1p*k_2p, {k_1p:k_1,k_2p:k_2}))

for i in range(100):
    _, cost_val, W_val = sess.run([train, cost, W], {x_:x_train, y_:y_train})
    print("===========")
    print("iterater: %s" % i)
    print("+++++++")
    print("loss: %s" % cost_val)
    print("+++++++")
    print("w: %s" % W_val)
    print("===========")

