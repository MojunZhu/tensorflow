'''
Created on Feb 25, 2017

@author: mozhu
'''
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMCell,\
    DropoutWrapper, MultiRNNCell
import datetime

def __init__():
    batch_size = 100 # the layer size how many words in each  
    num_steps = 100 # how many batches, training data size
    hidden_size = 100
    keep_prob = 0.6
    vocab_size = 4000
    max_grad_norm = 1

    lstm_cell = LSTMCell(hidden_size, forget_bias=0)
    lstm_cell = DropoutWrapper(lstm_cell, output_keep_prob = keep_prob)
    cell = MultiRNNCell([lstm_cell] * batch_size)

    input_data = tf.placeholder(tf.float32, [batch_size, num_steps])
    targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    
    # Here use batch_size, because initial_state is for a connected cell group. it looks like [hidden_size][batch_size]
    # Because cell is a connected, so, it should gives outer layer of size which is batch_size
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
    inputs = tf.nn.embedding_lookup(embedding, input_data)

    # dropout input 
    inputs = tf.nn.dropout(inputs, keep_prob)
    
    outputs = []
    state = initial_state
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            # run RNN from the state, output new state for cell
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)
    
    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
    softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b       
    
    # loss is average negative log probability here we have exist function sequence_loss_by_example
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(targets,[-1])], [tf.ones([batch_size * num_steps])])

    total_cost = tf.reduce_sum(loss)
    final_state = state
    
    learning_rate = tf.Variable(0.0, trainable = False)
    tvars = tf.trainable_variables()
    
    grads, _ = tf.clip_by_global_norm(tf.gradients(total_cost, tvars), max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    
def run_epoch(session, m, data, eval_op, verbose=False):
    # run the model on given data
    epoch_size = ((len(data) // m.batch_size) -1) // m.num_steps
    start_time = datetime.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    
    
    
    
    



    


