import numpy as np
import tensorflow as tf

'''
# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})

# evaluate training  accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
'''

'''
prepareData here we have x(i) = [1, 5]; y(i) = [1,1]
W = [5, 1], b = [1,1]
model => y = W*x + b
'''

# prepare data
W = tf.Variable(tf.random_normal([5,1], stddev = 0.35, mean=2), name = "weight")

y_ = tf.placeholder(tf.float32, [6,1], name = "target")
x_ = tf.placeholder(tf.float32, [6,5], name = "input_data")

linear_regression_model = tf.matmul(x_,W)

y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(linear_regression_model - y_), 0)


optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)

x_train = [[1,2,3,4,5], [3,4,5,6,7], [9,7,4,2,1], [7,4,1,6,5], [1,5,3,6,7], [1,4,5,7,3]]
W_ = [[1],[2],[3],[4],[5]]
y_train_k = np.dot(x_train, W_)
print(y_train_k)

# normalize data
x_train_normalize = np.divide(np.subtract(x_train, np.mean(x_train)), np.std(x_train))
print(x_train_normalize)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#print(sess.run(loss, {x_:x_train, W: W_, b: b_, y_:y_train_k}))


for i in range(100):
    _, loss_val, W_val = sess.run([train, loss, W], {x_:x_train_normalize, y_:y_train_k})
    print("===========")
    print("iterater: %s" % i)
    print("+++++++")
    print("loss: %s" % loss_val)
    print("+++++++")
    print("w: %s" % W_val)
    print("===========")








