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
from keras.layers import Dense, LSTM, Lambda, Input, TimeDistributed, concatenate
from keras import optimizers, backend as K
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session

VOCA_SIZE = 70000
EM_DIM = 1
TEXT_DIR = '../../data/news/'
STOCK_DIR = '../../data/stock.csv'
VALIDATION = 0.1
TITLE_LEN = 15
TEXT_LEN = 300

learning_rate = 0.01
hidden_dim1 = 10
hidden_dim2 = 10
hidden_dim3 = 10
bs = 100 
ne = 10

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
random.shuffle(total_data)
print('total:', len(total_data))
train_data = total_data[int(VALIDATION*len(total_data)):]
test_data = total_data[:int(VALIDATION*len(total_data))]

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
model_a = Sequential()
model_a.add(Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
        weights = [embedding_mat], input_length = TITLE_LEN, \
        trainable=True))
#model_a.add(LSTM(hidden_dim1, return_sequences=True))
#model_a.add(TimeDistributed(Dense(hidden_dim1, activation = 'tanh')))
input_a = Input(shape=(TITLE_LEN,))
processed_a = model_a(input_a)
pool_rnn_a = Lambda(lambda x: K.sum(x, axis=1))(processed_a)
#pool_rnn_a = Lambda(lambda x: K.sum(x, axis=1), output_shape = (hidden_dim1,))(processed_a)

model_b = Sequential()
model_b.add(Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
        weights = [embedding_mat], input_length = TEXT_LEN, \
        trainable=True))
#model_b.add(LSTM(hidden_dim2, return_sequences=True))
#model_b.add(TimeDistributed(Dense(hidden_dim2, activation = 'tanh')))
input_b = Input(shape=(TEXT_LEN,))
processed_b = model_b(input_b)
pool_rnn_b = Lambda(lambda x: K.sum(x, axis=1))(processed_b)
#pool_rnn_b = Lambda(lambda x: K.sum(x, axis=1), output_shape = (hidden_dim2,))(processed_b)

merged = concatenate([pool_rnn_a, pool_rnn_b])
print(merged.shape)
#fc1 = Dense(hidden_dim3, input_dim=hidden_dim1+hidden_dim2, activation='softmax')(merged)
#output = Dense(1, input_dim=hidden_dim3, activation='softmax')(fc1)
output = Dense(1, input_dim=2, activation='softmax')(merged)

final_model = Model(inputs = [input_a, input_b], outputs = output)
adam = optimizers.Adam(lr=learning_rate)
final_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

final_model.summary()

#early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

final_model.fit([train_x1, train_x2], train_y, batch_size = bs, epochs = ne, \
	verbose=1, validation_data=([test_x1, test_x2], test_y))#, callbacks=[early_stop])
