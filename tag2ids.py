tags = []
tag2id = {'b':'1', 'm': '2', 'e':'3', 's':'4'}
with open('tags', 'r') as f:
    for line in f.readlines():
        l = []
        l.extend(line.strip())
        tags.append([tag2id[t] for t in l])
with open('tag2id', 'w') as f:
    for tag_line in tags:
        if len(tag_line) < 32:
            tag_line = tag_line + (['0'] * (32 - len(tag_line)))
        else:
            tag_line = tag_line[:32]
        f.write(' '.join(tag_line))
        f.write('\n')
