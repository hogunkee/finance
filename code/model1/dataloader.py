import os
import numpy as np
import datetime
import pandas
import random
from scipy import stats
from keras.preprocessing.text import Tokenizer

TEXT_DIR = '../data/news/'
STOCK_DIR = '../data/stock.csv'

## load news data ##
def load_text(text_dir):
    texts = []
    titles = []
    dates = []
    for name in sorted(os.listdir(text_dir)):
        path = os.path.join(text_dir, name)
        if os.path.isdir(path):
            for fname in os.listdir(path):
                fpath = os.path.join(path, fname)
                f = open(fpath, 'r')
                texts.append(f.read())
                titles.append(fname)
                dates.append(name)
    return texts, titles, dates

## load stock data ##
def split_date_stock(line):
    date = line.split(',')[0]
    #date = list(map(lambda k: int(k), line.split(',')[0].split('-')))
    stock = int(line.split(',')[1].replace('\n',''))
    return date, stock

def load_stock(stock_dir):
    f = open(stock_dir, 'r')
    lines = f.readlines()[1:]
    lines = sorted(list(map(lambda k: split_date_stock(k), lines)), key=lambda q: q[0])
    date, stock = zip(*lines)
    return date, stock

def get_stock_slope(date, stock):
    # input date form: '2015-01-01'
    assert len(date)==len(stock)
    tslope = [0 for _ in range(len(date)-4)]
    split_date = list(map(lambda x: datetime.date(x[0],x[1],x[2]), \
            list(map(lambda d: list(map(lambda k: int(k), d.split('-'))), date))))
    for i in range(len(date)-4):
        x = list(map(lambda d: (d-split_date[i]).days, split_date[i:i+5]))
        y = list(map(lambda s: 100*s/stock[i], stock[i: i+5]))
        tslope[i] = stats.linregress(x,y)[0]

    c = 0
    slope = []
    updown = []
    daterange = pandas.date_range(date[0], date[-1])
    wholedate = list(map(lambda d: str(d).split(' ')[0], daterange))
    for _date in wholedate:
        if (date[c] < _date):
            c+=1
            if c>=len(tslope):
                break
        slope.append(tslope[c])
        updown.append(1 if tslope[c]>0 else 0)

    return list(zip(slope, wholedate[:len(slope)+1], updown))

def stock_data(stock_dir):
    date, stock = load_stock(stock_dir)
    return get_stock_slope(date, stock)

## embedding mat ##
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
