# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
import os
import pickle
import numpy as np
from dataloader import load_text, stock_data, make_embedding

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, LSTM, Lambda, Input
from keras.preprocessing.sequence import pad_sequences

VOCA_SIZE = 500000
TEXT_DIR = '../data/news/'
STOCK_DIR = '../data/stock.csv'
validation = 0.1

def save_obj(obj, name):
    with open('obj/'+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)

def exist(name):
    return os.path.exists('obj/'+name+'.pkl')

if exist('word2index') and exist('index2word') and exist('embedding_mat'):
    tokenizer = load_obj('tokenizer')
    word2index = load_obj('word2index')
    index2word = load_obj('index2word')
    embedding_mat = load_obj('embedding_mat')
else:
    tokenizer, word2index, index2word, embedding_mat = make_embedding(TEXT_DIR, VOCA_SIZE)
    save_obj(tokenizer, 'tokenizer')
    save_obj(word2index, 'word2index')
    save_obj(index2word, 'index2word')
    save_obj(embedding_mat, 'embedding_mat')

if exist('total_data'):
    total_data = load_obj('total_data')
else:
    texts, titles, dates = load_text(TEXT_DIR)
    titles_seq = tokenizer.texts_to_sequences(titles)
    texts_seq = tokenizer.texts_to_sequences(texts)

    stock, sdate, updown = zip(*stock_data(STOCK_DIR))
    date2updown = dict(zip(sdate, updown))
    labels = list(map(lambda d: date2updown[d], dates))
    total_data = list(zip(titles_seq, texts_seq, labels))
    save_obj(total_data, 'total_data')

# data format: x1(title), x2(text), y(up/down label: 1 or 0)
sorted_data = sorted(total_data)
train_data = sorted_data[int(validation*len(total_data)):]
test_data = sorted_data[:int(validation*len(total_data))]


