# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
import os 
import pickle
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from dataloader import load_text, stock_data, make_embedding

from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Reshape, Flatten, Add
from keras.layers import Dropout, Concatenate, BatchNormalization, LeakyReLU
from keras import optimizers, backend as K
from keras.regularizers import l2 as L2
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session

VOCA_SIZE = 70000
EM_DIM = 300
TEXT_DIR = '../../data/news/'
STOCK_DIR = '../../data/stock.csv'
VALIDATION = 0.1
TITLE_LEN = 15
TEXT_LEN = 300
NUM_INPUT = 20
GEN_RANDOM = 10
GEN_RANDOM_TEST = 100
#GEN_RANDOM = 100

test_sample = 30
learning_rate = 0.01
beta = 1e-5
hidden_dim1 = 200
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5
bs = 100
ne = 5

def save_obj(obj, name):
    with open('obj/'+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)

def exist(name):
    return os.path.exists('obj/'+name+'.pkl')

def sampling(data, gen_random):
    sampled_texts = []
    sampled_labels = []
    titles, texts, labels = zip(*data)
    for x in range(len(titles)):
        title = titles[x]
        text = [t[:100] for t in texts[x]]
        label = labels[x]
        for y in range(gen_random):
            idx = np.random.choice(len(title), NUM_INPUT)
            sampled_texts.append(' '.join(title+text))
            #sampled_texts.append(' '.join([text[i] for i in idx]))
            sampled_labels.append(label)
    assert len(sampled_texts)==len(sampled_labels)
    return list(zip(sampled_texts, sampled_labels))


if exist('total_data'):
    print('import total data')
    total_data = load_obj('total_data')
else:
    texts, titles, dates = load_text(TEXT_DIR)
    assert len(texts) == len(dates)

    stock, sdate, updown = zip(*stock_data(STOCK_DIR))
    date2updown = dict(zip(sdate, updown))
    labels = list(map(lambda d: date2updown[d], dates))
    total_data = np.array(list(zip(titles, texts, labels)))
    save_obj(total_data, 'total_data')


# data format: x1(title), x2(text), y(up/down label: 1 or 0)
print('total:', len(total_data))
train_data = total_data[:-test_sample]
test_data = total_data[-test_sample:]

train_data = sampling(train_data, GEN_RANDOM)
test_data = sampling(test_data, GEN_RANDOM_TEST)
random.shuffle(train_data)

train_x, train_y = zip(*train_data)
test_x, test_y = zip(*test_data)

train_x = list(train_x)
test_x = list(test_x)
train_y = list(train_y)
test_y = list(test_y)
assert len(test_x)==len(test_y)

'''
test_x = [test_x1[i]+test_x2[i] for i in range(len(test_x1))]
train_x = list(map(np.array, zip(*train_x)))
test_x = list(map(np.array, zip(*test_x)))

train_y = np.array(train_y)
test_y = np.array(test_y)
'''

def main():
    basicvectorizer = CountVectorizer(ngram_range=(2,2))
    basictrain = basicvectorizer.fit_transform(train_x)

    basicmodel = LogisticRegression()
    basicmodel = basicmodel.fit(basictrain, train_y)

    basictest = basicvectorizer.transform(test_x)
    predictions = basicmodel.predict(basictest)

    pd.crosstab([test_y], [predictions], rownames=["Actual"], colnames=["Predicted"])

    print(basictrain.shape)

    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix

    print (classification_report(test_y, predictions))
    print (accuracy_score(test_y, predictions))

if __name__=='__main__':
    main()
