import os
import numpy as np
from pprint import pprint
import random
from dataloader import load_text
from keras.preprocessing.text import Tokenizer

def make_embedding(text_dir, voca_size):
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

    texts, titles, datelist = load_text(text_dir)
    tokenizer = Tokenizer(voca_size)
    tokenizer.fit_on_texts(texts + titles)
    word2index = tokenizer.word_index
    index2word = {v: k for k,v in word2index.items()}

    ## new embedding
    embedding_mat = np.zeros((voca_size, embedding_dim))
    for _word, _idx in word2index.items():
        if _idx < voca_size:
            if _word in pre_embedding_word.keys():
                embedding_mat[_idx] = pre_embedding_word[_word]
            else:
                embedding_mat[_idx] = np.random.normal(0, 0.1, embedding_dim)

    return tokenizer, word2index, index2word, embedding_mat
