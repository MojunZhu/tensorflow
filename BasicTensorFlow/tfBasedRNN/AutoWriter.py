'''
Created on Mar 22, 2017

@author: mozhu
'''
from idlelib.IOBinding import encoding
import collections
import numpy as np
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import BasicLSTMCell,\
    DropoutWrapper, MultiRNNCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import sequence_loss_by_example
import tensorflow as tf

# build vocab index dict
vocab = {}
vocabSize = 1000

with open("C:\\Users\\mozhu\\Desktop\\sansheng.txt", 'r', encoding='utf-8') as f:
    while True:
        c = f.read(1)
        if not c:
            print("end of file")
            break
        if c in vocab:
            vocab[c] += 1
        else:
            vocab.setdefault(c, 1)

selected_vocab = collections.Counter(vocab).most_common(vocabSize - 1)
print(selected_vocab)

index2vocab = {0: 'unknow_words'}
vocab2index = {'unknow_words': 0}
index = 1;
for k, v in selected_vocab:
    index2vocab[index] = k
    vocab2index[k] = index
    index += 1

# build indexed input
num_steps = 32

input_indexed_x = np.empty((1,num_steps))
with open("C:\\Users\\mozhu\\Desktop\\sansheng.txt", 'r', encoding='utf-8') as f:
    while True:
        c = f.read(num_steps)
        current_row = np.zeros((1,num_steps))
        if not c:
            print("end of file")
            break
        c_list = list(c)
        c_index = 0
        for word in c_list:
            if word in vocab2index:
                current_row[0,c_index] = vocab2index[word]
                c_index+=1
            else:
                c_index+=1
                continue
        #print(current_row)
        input_indexed_x = np.append(input_indexed_x, current_row, axis = 0)
        #print(input_indexed_x)
           
print(input_indexed_x.shape)
print(input_indexed_x[0,:])

# build indexed input which is one character after input
input_indexed_y = np.empty((1,num_steps))
with open("C:\\Users\\mozhu\\Desktop\\sansheng.txt", 'r', encoding='utf-8') as f:
    f.read(1)
    while True:
        c = f.read(num_steps)
        current_row = np.zeros((1,num_steps))
        if not c:
            print("end of file")
            break
        c_list = list(c)
        c_index = 0
        for word in c_list:
            if word in vocab2index:
                current_row[0,c_index] = vocab2index[word]
                c_index+=1
            else:
                c_index+=1
                continue
        #print(current_row)
        input_indexed_y = np.append(input_indexed_x, current_row, axis = 0)
        #print(input_indexed_x)
        
print(input_indexed_y.shape)
print(input_indexed_y[0,:])

test_x = input_indexed_x[5000:5004]
for i1 in test_x:
    for j1 in i1:
        print(index2vocab[j1], end="")
    print("\n")
input_indexed_x = input_indexed_x[:1000]
input_indexed_y = input_indexed_y[1:1001]

# here we build a multi-layer RNN with LSTM
num_of_steps = 32
hidden_state_size = 20
rnn_layer = 1
learn_rate = 0.001
epoch_size = 20

# prepare test data
test_input_x = test_x 
input_x = input_indexed_x
input_target = input_indexed_y

# setup variables for graph
x_ = tf.placeholder(tf.int32, [None, num_of_steps])
y_ = tf.placeholder(tf.int32, [None, num_of_steps])

batch_size = tf.shape(x_)[0] # get batch size 


lstm_cell = BasicLSTMCell(hidden_state_size, forget_bias=0.5, state_is_tuple = True)
attn_cell = DropoutWrapper(lstm_cell, output_keep_prob=1) 
cell = MultiRNNCell([attn_cell for _ in range(rnn_layer)], state_is_tuple = True)
initial_state = cell.zero_state(batch_size, tf.float32)

# turn our x_placeholder to one-hot tensor
x_one_hot_ = tf.one_hot(x_, vocabSize)
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
    W = tf.get_variable("W", [hidden_state_size, vocabSize])
    b = tf.get_variable("b", [vocabSize], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in tf.unstack(rnn_outputs, axis = 1)] # logits = [steps, batch_size, vacab]
predictions = [tf.nn.softmax(logit) for logit in logits] #[steps, batch, vocab]
# print(predictions)

'''
 why use logists as cost, is because logists can provide semi-one-hot vector with each entry some value. Then with one-hot y, all values
 are error except y's 1 entry, before softmax, it is kind of continues, softmax makes it not continue
'''
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

# predicted result
for i in range(epoch_size):
    _, cost_val, w_val, b_val, final_state_val = sess.run([train_step, total_loss, W, b, final_state], {x_:input_x, y_:input_target})
    print("===========")
    print("iterater: %s" % i)
    print("+++++++")
    print("loss: %s" % cost_val)
    print("+++++++")
    print("w_val %s" % w_val)
    print("+++++++")
    print("final_val %s" % final_state_val)
    print("===========")
 

predictions_val = sess.run([predictions], {x_:test_input_x})
predictions_val = np.asarray(predictions_val[0])
predictions_k = tf.argmax(predictions_val, axis=2)
predictions_k = sess.run(predictions_k)
predictions_k = np.asarray(predictions_k)
print(predictions_k)

# convert back to words
predictions_k = np.transpose(predictions_k)

for i in predictions_k:
    for j in i:
        print(index2vocab[j], end="")
    print("\n")

