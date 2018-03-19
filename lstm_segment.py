#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import time

decay = 0.85
max_epoch = 5
max_max_epoch = 1

timestep_size = max_len = 32
vocab_size = 10000
input_size = embedding_size = 64
class_num = 5
hidden_size = 128
lay_num = 2
max_grad_norm = 5.0

lr = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
batch_size = tf.placeholder(tf.int32, shape=[])
model_save_path = 'ckpt/bi-lstm.ckpt'

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

X_inputs = tf.placeholder(tf.int32, [None, timestep_size], name = 'X_input')
y_inputs = tf.placeholder(tf.int32, [None, timestep_size], name = 'y_inpput')

def lstm(X_inputs):
    embedding = tf.get_variable('embedding', [vocab_size, embedding_size], dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, X_inputs)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    output, state = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state = initial_state, dtype=tf.float32)
    
    output = tf.reshape(output, shape=[-1, hidden_size])
    w  = weight_variable([hidden_size, class_num])
    b = bias_variable([class_num])
    logits = tf.matmul(output, w) + b
    return logits

y_pred = lstm(X_inputs)
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y_inputs, [-1]), logits=y_pred))
tf.add_to_collection('predict', y_pred)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
train_op = optimizer.apply_gradients(zip(grads, tvars), global_step = tf.contrib.framework.get_or_create_global_step())

correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print 'Finished creating the lstm model'

#-------------------------------------------------------------
#traing
X = []
with open('X') as f:
    for line in f.readlines():
        X.append(line.strip().split(' '))
Y = []
with open('Y') as f:
    for line in f.readlines():
        Y.append(line.strip().split(' '))

data_train = X[:-10000]
label_train = Y[:-10000]
data_test = X[-10000:]
label_test = Y[-10000:]
data_train_len = len(data_train)

global_index = 0
def next_batch(size):
    global global_index
    data_batch = []
    label_batch = []
    for i in xrange(size):
        if global_index + i < data_train_len:
            data_batch.append(data_train[global_index + i])
            label_batch.append(label_train[global_index + i])
        else:
            data_batch.append(data_train[(global_index + i) % data_train_len])
            label_batch.append(label_train[(global_index + i) % data_train_len])
    global_index += size
    return data_batch, label_batch
        

tr_batch_size = 128
tr_batch_num = data_train_len / tr_batch_size
saver = tf.train.Saver(max_to_keep=10)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in xrange(max_max_epoch):
        _lr = 1e-4
        if epoch > max_epoch:
            _lr = _lr * ((decay) ** (epoch-max_epoch))
        print 'EPOCH %d， lr=%g' % (epoch+1, _lr)
        start_time = time.time()
        _costs = 0.0
        _accs = 0.0
        show_accs = 0.0
        show_costs = 0.0
        for batch in xrange(tr_batch_num):
            fetches = [accuracy, cost, train_op]
            X_batch, y_batch = next_batch(tr_batch_size)
            _acc, _cost, _ = sess.run(fetches, feed_dict={X_inputs:X_batch, y_inputs:y_batch, lr:_lr, batch_size:tr_batch_size, keep_prob:0.5})
            _accs += _acc
            _costs += _cost
            show_accs += _acc
            show_costs += _cost
        save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
        mean_acc = _accs / tr_batch_num
        mean_cost = _costs / tr_batch_num
        print 'acc=%g, cost=%g ' % (mean_acc, mean_cost)

#--------------------------------------------------------------
# testing

#print '**Test Result:'
#test_acc, test_cost = test_epoch(data_test)
#print '**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost)    
