import json
with open('word_ids', 'r') as f:
    word_ids = json.load(f)

words = []
with open('words', 'r') as f:
    for line in f.readlines():
        l = []
        l.extend(line.strip())
        words.append([str(word_ids[w]) if w in word_ids else '0' for w in l])

with open('word2id', 'w') as f:
    for word_line in words:
        if len(word_line ) < 32:
            word_line = word_line + ['0'] * (32 - len(word_line))
        else:
            word_line = word_line[:32]
        f.write(' '.join(word_line))
        f.write('\n')

