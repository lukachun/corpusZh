#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re

TAG=['b','m', 'e', 's']
word_list=[]
with open('example.txt', 'r') as f:
    for line in f.readlines():
        cols = [re.sub(r'/.*$', '', w)for w in line.strip().split(' ')]
        text = ' '.join(cols)
        cols = re.split(r'[；，。]', text)
        for col in cols:
            if col:
                word_list.append(col.strip())

words = []
tags = []
for wl in word_list:
    word_for_line = []
    tag_for_line = []
    cols = wl.split(' ')
    for col in cols:
        if len(col) == 0:
            continue
        elif len(col) == 1:
            word_for_line.extend(col)
            tag_for_line.extend('s')
        elif len(col) == 2:
            word_for_line.extend(col)
            tag_for_line.extend('be')
        else:
            word_for_line.extend(col)
            tmp_tag = 'm' * (len(col) - 2)
            tag_for_line.extend('b' + tmp_tag + 'e')
    words.append(word_for_line)
    tags.append(tag_for_line)

with open('words', 'w') as f:
    for wl in words: 
        f.write(''.join(wl))
        f.write('\n')

with open('tags', 'w') as f:
    for tl in tags: 
        f.write(''.join(tl))
        f.write('\n')

