import os
import numpy as np
from pprint import pprint
import random

TEXT_DIR = '../sample/'

## read pre-trained embeddings (fasttext)
'''
embedding_word = {}
f = open('../data/wiki.ko.vec', 'r')
lines = f.readlines()
print('num of embedding lines: %d' %len(lines))
for i in range(1000):
    values = lines[i].split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_word[word] = coefs
    enbedding_dim = len(values[1:])
f.close()
'''
#print(embedding_word)

## word counting
NUM_SAMPLE = 50000
count_word = {}
filelist = []
for date in os.listdir(TEXT_DIR):
    datepath = os.path.join(TEXT_DIR, date)
    filelist += list(map(lambda k: os.path.join(datepath, k), os.listdir(datepath)))

for filepath in random.sample(filelist, NUM_SAMPLE):
    _title = filepath.split('/')[-1]

    f = open(filepath, 'r')
    _text = f.read()
    whole_text = _title + ' ' + _text
    words = whole_text.split()

    for _word in words:
        if not _word.isdigit():
            if _word in count_word.keys():
                count_word[_word] += 1
            else:
                count_word[_word] = 1
most_words = sorted(list(count_word.items()), key=lambda x: x[1], reverse=True)

pprint(most_words[:100])
print(len(most_words))

