import os
import numpy as np
from pprint import pprint
import random

VOCA_SIZE = 200000
TEXT_DIR = '../sample/'
NUM_SAMPLE = 200000

## read pre-trained embeddings (fasttext)
pre_embedding_word = {}
f = open('../data/wiki.ko.vec', 'r')
lines = f.seek(11)
lines = f.readlines()
print('num of embedding lines: %d' %len(lines))

for i in range(len(lines)):
    values = lines[i].split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    pre_embedding_word[word] = coefs
embedding_dim = len(coefs)
f.close()
#print(pre_embedding_word)

## word counting
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
sorted_count = sorted(list(count_word.items()), key=lambda x: x[1], reverse=True)
most_words = list(zip(*sorted_count))[0]
word2index = dict(list(zip(most_words, [i+1 for i in range(len(most_words))])))
index2word = {v: k for k,v in word2index.items()}

#pprint(most_words[:100])
print('most words', len(most_words))

## new embedding
'''
embedding_word = {}
no_dic = {}
for _word in most_words:
    if _word in pre_embedding_word.keys():
        embedding_word[_word] = pre_embedding_word[_word]
    else:
        no_dic[_word] = 0
'''
embedding_mat = np.zeros((VOCA_SIZE, embedding_dim))
for _word, _idx in word2index.items():
    if _idx < VOCA_SIZE:
        if _word in pre_embedding_word.keys():
            embedding_mat[_idx] = pre_embedding_word[_word]
        else:
            embedding_mat[_idx] = np.random.normal(0, 0.1, embedding_dim)
