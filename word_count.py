# -*- coding: utf-8 -*-

from collections import Counter
import json

with open('words', 'r') as f:
    words = f.read()
c = Counter(words)

dic = {}
mc = c.most_common(10000)
mc[0] = ('UNK', 0)

wid = 0
for co in mc:
    dic[co[0]] = wid
    wid += 1

with open('word_ids', 'w') as f:
    json.dump(dic, f)
