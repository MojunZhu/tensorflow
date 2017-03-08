'''
Created on Mar 3, 2017

@author: mozhu
'''
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import BasicLSTMCell,\
    DropoutWrapper, MultiRNNCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import sequence_loss_by_example

'''
Here we define vacab_size = 60; batch_size = 10;
hidden_state_size(num_units in one cell) = 20; num_of_steps = 32; layer = 1
'''
import tensorflow as tf
import numpy as np

# here we build a multi-layer RNN with LSTM
vocab_size = 60
batch_size = 10
num_of_steps = 32
hidden_state_size = 20
rnn_layer = 1
learn_rate = 0.01
epoch_size = 10

# prepare test data
input_x = np.random.randint(59, size = (batch_size, num_of_steps))
input_target = np.random.randint(59, size = (batch_size, num_of_steps))

# setup variables for graph
x_ = tf.placeholder(tf.int32, [batch_size, num_of_steps])
y_ = tf.placeholder(tf.int32, [batch_size, num_of_steps])


lstm_cell = BasicLSTMCell(hidden_state_size, forget_bias=0.5, state_is_tuple = True)
attn_cell = DropoutWrapper(lstm_cell, output_keep_prob=1) 
cell = MultiRNNCell([attn_cell for _ in range(rnn_layer)], state_is_tuple = True)
initial_state = cell.zero_state(batch_size, tf.float32)

# turn our x_placeholder to one-hot tensor
x_one_hot_ = tf.one_hot(x_, vocab_size)
# rnn_inputs = tf.unstack(x_one_hot_, axis=1)

'''
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(x_one_hot_,{x_:input_x})

print(x_)
print(x_one_hot_)
print(rnn_inputs)
'''

rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, x_one_hot_, initial_state=initial_state, dtype=tf.float32)

# print(rnn_outputs)  # outputs = [batchsize, steps, hidden_state_size] outputs are all hidden state for one epoch
'''
Predictions, loss, training steps
'''
with tf.variable_scope("softmax"):
    W = tf.get_variable("W", [hidden_state_size, vocab_size])
    b = tf.get_variable("b", [vocab_size], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in tf.unstack(rnn_outputs, axis = 1)] # logits = [steps, batch_size, vacab]
predictions = [tf.nn.softmax(logit) for logit in logits]
# print(predictions)

y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(y_, num_of_steps, axis=1)] # y_as_list = [steps, batch]
# print(y_as_list)
# print(y_one_hot_)

loss_weights = [tf.ones([batch_size]) for i in range(num_of_steps)]
# print(loss_weights)
losses = sequence_loss_by_example(logits, y_as_list, loss_weights) # this is calculated step by step so, step should go as first index
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learn_rate).minimize(total_loss)

# run prediction
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(epoch_size):
    _, cost_val= sess.run([train_step, total_loss], {x_:input_x, y_:input_target})
    print("===========")
    print("iterater: %s" % i)
    print("+++++++")
    print("loss: %s" % cost_val)
    print("===========")
 
