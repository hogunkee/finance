# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
import os 
import pickle
import random
import numpy as np
import tensorflow as tf
from dataloader import load_text, stock_data, make_embedding

from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Reshape, Flatten
from keras.layers import Dropout, Concatenate, BatchNormalization, LeakyReLU
from keras import optimizers, backend as K
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session

VOCA_SIZE = 70000
EM_DIM = 1
TEXT_DIR = 'home/qara/hogun/data/news/'
STOCK_DIR = 'home/qara/hogun/data/stock.csv'
VALIDATION = 0.1
TITLE_LEN = 15
TEXT_LEN = 300

test_sample = 12584
learning_rate = 0.01
hidden_dim1 = 200
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5
bs = 500 
ne = 5

def save_obj(obj, name):
    with open('obj/'+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)

def exist(name):
    return os.path.exists('obj/'+name+'.pkl')

if exist('word2index') and exist('index2word') and exist('embedding_mat'):
    print('load word2index, index2word, embedding matrix')
    tokenizer = load_obj('tokenizer')
    word2index = load_obj('word2index')
    index2word = load_obj('index2word')
    embedding_mat = load_obj('embedding_mat')
else:
    tokenizer, word2index, index2word, embedding_mat = make_embedding(TEXT_DIR, VOCA_SIZE, EM_DIM)
    save_obj(tokenizer, 'tokenizer')
    save_obj(word2index, 'word2index')
    save_obj(index2word, 'index2word')
    save_obj(embedding_mat, 'embedding_mat')

if exist('total_data'):
    print('import total data')
    total_data = load_obj('total_data')
else:
    texts, titles, dates = load_text(TEXT_DIR)
    titles_seq = pad_sequences(tokenizer.texts_to_sequences(titles), maxlen=TITLE_LEN)
    texts_seq = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=TEXT_LEN)

    stock, sdate, updown = zip(*stock_data(STOCK_DIR))
    date2updown = dict(zip(sdate, updown))
    labels = list(map(lambda d: date2updown[d], dates))
    total_data = np.array(list(zip(titles_seq, texts_seq, labels)))
    save_obj(total_data, 'total_data')

n_symbols = min(VOCA_SIZE, len(word2index))
print(n_symbols)

# data format: x1(title), x2(text), y(up/down label: 1 or 0)
print('total:', len(total_data))
train_data = total_data[:-test_sample]
test_data = total_data[-test_sample:]
random.shuffle(train_data)

train_x1, train_x2, train_y = zip(*train_data)
test_x1, test_x2, test_y = zip(*test_data)
train_x1 = np.array(train_x1)
train_x2 = np.array(train_x2)
train_y = np.array(train_y)
test_x1 = np.array(test_x1)
test_x2 = np.array(test_x2)
test_y = np.array(test_y)

K.clear_session()
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#config.log_device_placement = True
sess=tf.Session(config=config)
set_session(sess)

print(embedding_mat.shape)

input_a = Input(shape=(TITLE_LEN,))
embed_a = Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
        weights = [embedding_mat], input_length = TITLE_LEN, \
        trainable=True)(input_a)
reshape_a = Reshape((TITLE_LEN, EM_DIM, 1))(embed_a)
bn_a = BatchNormalization()(reshape_a)

conv_a0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], EM_DIM), \
        padding='valid', kernel_initializer='normal', activation='relu')(bn_a)
conv_a1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], EM_DIM), \
        padding='valid', kernel_initializer='normal', activation='relu')(bn_a)
conv_a2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], EM_DIM), \
        padding='valid', kernel_initializer='normal', activation='relu')(bn_a)

maxpool_a0 = MaxPool2D(pool_size=(TITLE_LEN - filter_sizes[0] + 1, 1), \
        strides=(1,1), padding='valid')(conv_a0)
maxpool_a1 = MaxPool2D(pool_size=(TITLE_LEN - filter_sizes[1] + 1, 1), \
        strides=(1,1), padding='valid')(conv_a1)
maxpool_a2 = MaxPool2D(pool_size=(TITLE_LEN - filter_sizes[2] + 1, 1), \
        strides=(1,1), padding='valid')(conv_a2)

input_b = Input(shape=(TEXT_LEN,))
embed_b = Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
        weights = [embedding_mat], input_length = TEXT_LEN, \
        trainable=True)(input_b)
reshape_b = Reshape((TEXT_LEN, EM_DIM, 1))(embed_b)
bn_b = BatchNormalization()(reshape_b)

conv_b0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], EM_DIM), \
        padding='valid', kernel_initializer='normal', activation='relu')(bn_b)
conv_b1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], EM_DIM), \
        padding='valid', kernel_initializer='normal', activation='relu')(bn_b)
conv_b2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], EM_DIM), \
        padding='valid', kernel_initializer='normal', activation='relu')(bn_b)

maxpool_b0 = MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[0] + 1, 1), \
        strides=(1,1), padding='valid')(conv_b0)
maxpool_b1 = MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[1] + 1, 1), \
        strides=(1,1), padding='valid')(conv_b1)
maxpool_b2 = MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[2] + 1, 1), \
        strides=(1,1), padding='valid')(conv_b2)

concatenated_tensor = Concatenate(axis=1)([maxpool_a0, maxpool_a1, maxpool_a2,\
        maxpool_b0, maxpool_b1, maxpool_b2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
fc = Dense(hidden_dim1)(dropout)
bn2 = BatchNormalization()(fc)
relu = LeakyReLU(0.3)(bn2)
output = Dense(1, activation='sigmoid')(relu)

final_model = Model(inputs = [input_a, input_b], outputs = output)
adam = optimizers.Adam(lr=learning_rate)
final_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

final_model.summary()

#early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

final_model.fit([train_x1, train_x2], train_y, batch_size = bs, epochs = ne, \
	verbose=1, validation_data=([test_x1, test_x2], test_y))#, callbacks=[early_stop])
