#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import json
import numpy as np

with open('word_ids', 'r') as f:
    word_ids = json.load(f)

s = u'并且保留了前面一步各个选择的最优解'
text_len = len(s)
word_list = []
word_list.extend(s)
word2id = [word_ids[w] for w in word_list]
word2id.extend([0] * (32-len(word2id)))
word2id = np.array(word2id)
word2id = np.expand_dims(word2id, axis=0)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('ckpt/bi-lstm.ckpt-3.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./ckpt'))
    graph = tf.get_default_graph()
    y_pred = tf.get_collection('y_pred')[0]
    X_input = graph.get_operation_by_name('X_input').outputs[0]
    batch_size = graph.get_operation_by_name('Placeholder_2').outputs[0]
    
    pred = sess.run(y_pred, feed_dict={X_input:word2id, batch_size:1})
    #print sess.run(tf.argmax(pred,1))
    print pred

T = {'4':'s', '1':'b', '2':'m', '3':'e'}
labels = []
with open('Y') as f:
    for line in f.readlines():
        cols = line.strip().split(' ')
        l = []
        for c in cols:
            if c == '0':
                break
            else:
                l.append(T[c])
        labels.append(l)

A = {
      'sb':0,
      'ss':0,
      'be':0,
      'bm':0,
      'me':0,
      'mm':0,
      'eb':0,
      'es':0
     }

START = {'b':0, 's':0}
END = {'e':0, 's':0}

for label in labels:
    for t in xrange(len(label)):
        if t == 0:
            START[label[t]] += 1.0
        elif t == len(label) - 1:
            if label[t] in END:
                END[label[t]] += 1.0
        else:
            key = label[t] + label[t+1]
            A[key] += 1.0
zy = dict()
zy['sb'] = A['sb'] / (A['sb'] + A['ss'])
zy['ss'] = 1.0 - zy['sb']
zy['be'] = A['be'] / (A['be'] + A['bm'])
zy['bm'] = 1.0 - zy['be']
zy['me'] = A['me'] / (A['me'] + A['mm'])
zy['mm'] = 1.0 - zy['me']
zy['eb'] = A['eb'] / (A['eb'] + A['es'])
zy['es'] = 1.0 - zy['eb']

keys = sorted(zy.keys())

zy = {key:np.log(value) for key, value in zy.items()}

zs = dict()
zs['b'] = START['b'] / (START['b'] + START['s'])
zs['s'] = 1.0 - zs['b']
zs = {key:np.log(value) for key, value in zs.items()}
print zs

ze = dict()
ze['e'] = END['e'] / (END['e'] + END['s'])
ze['s'] = 1.0 - ze['e']
ze = {key:np.log(value) for key, value in ze.items()}
print ze


N_INF = float('-inf')
pred = pred[:text_len, 1:]
tag_list = ['b', 'm', 'e', 's']
states = [0, 1, 2, 3]
path = {s:[] for s in states}
def veterbi(nodes):
    print 'nodes:', nodes
    start = nodes[0]
    t = [zs.get(tag_list[i], N_INF) for i in states]
    curr_prob = {i:start[i] + t[i] for i in states}
    last_prob = {}
    for layer in nodes[1:]:
        last_prob = curr_prob
        curr_prob = {}
        for curr_state in states:
            max_prob, last_stat = max(((last_prob[last_state] + zy.get(tag_list[last_state] + tag_list[curr_state], N_INF) + layer[curr_state], last_state) for last_state in states))
            curr_prob[curr_state] = max_prob
            path[curr_state].append(last_stat)
    reverse_path = []
    print 'path:', path
    max_pro = -1
    max_path = None
    last_s = None
    for s in states:
        if curr_prob[s] > max_pro:
            last_s = s
            max_pro = curr_prob[s]
    reverse_path.append(last_s)
    for i in reversed(range(len(path[s]))):
        reverse_path.append(path[last_s][i])
        last_s = path[last_s][i]
    return reversed(reverse_path)
tag_index = veterbi(pred)
tag_pred = [tag_list[tag] for tag in tag_index]

#---------------------------------------------
result = ''
i = 0
for tag in tag_pred:
    if tag == 's':
        result += word_list[i] + '/'
    elif tag == 'b' or tag == 'm':
        result += word_list[i]
    else:
        result += word_list[i] + '/'
    i += 1
print result
#---------------------------------------------
