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
    sampled_titles = []
    sampled_texts = []
    sampled_labels = []
    titles, texts, labels = zip(*data)
    for x in range(len(titles)):
        title = titles[x]
        text = texts[x]
        label = labels[x]
        for y in range(gen_random):
            idx = np.random.choice(len(title), NUM_INPUT)
            sampled_titles.append([title[i] for i in idx])
            sampled_texts.append([text[i] for i in idx])
            sampled_labels.append(label)
    assert len(sampled_titles)==len(sampled_labels)
    return list(zip(sampled_titles, sampled_texts, sampled_labels))


if exist('word2index') and exist('index2word') and exist('embedding_mat'):
    print('load word2index, index2word, embedding matrix')
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
    print('import total data')
    total_data = load_obj('total_data')
else:
    texts, titles, dates = load_text(TEXT_DIR)
    assert len(texts) == len(dates)

    titles_seq = []
    texts_seq = []
    for t in titles:
        t_seq = pad_sequences(tokenizer.texts_to_sequences(t), maxlen=TITLE_LEN)
        titles_seq.append(t_seq)
    for t in texts:
        t_seq = pad_sequences(tokenizer.texts_to_sequences(t), maxlen=TEXT_LEN)
        texts_seq.append(t_seq)

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

train_data = sampling(train_data, GEN_RANDOM)
test_data = sampling(test_data, GEN_RANDOM_TEST)
random.shuffle(train_data)

train_x1, train_x2, train_y = zip(*train_data)
test_x1, test_x2, test_y = zip(*test_data)

train_x = [train_x1[i]+train_x2[i] for i in range(len(train_x1))]
test_x = [test_x1[i]+test_x2[i] for i in range(len(test_x1))]
train_x = list(map(np.array, zip(*train_x)))
test_x = list(map(np.array, zip(*test_x)))

train_y = np.array(train_y)
test_y = np.array(test_y)

def main():

    K.clear_session()
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    #config.log_device_placement = True
    sess=tf.Session(config=config)
    set_session(sess)

    print(embedding_mat.shape)

    inputA = [Input(shape=(TITLE_LEN,)) for k in range(NUM_INPUT)]
    inputB = [Input(shape=(TEXT_LEN,)) for k in range(NUM_INPUT)]
    maxpoolA_1, maxpoolA_2, maxpoolA_3, maxpoolB_1, maxpoolB_2, maxpoolB_3=[[] for i in  range(6)]

    modelA_in = Sequential()
    modelA_in.add(Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
            weights = [embedding_mat], input_length = TITLE_LEN, \
            trainable=True))
    modelA_in.add(Reshape((TITLE_LEN, EM_DIM, 1)))

    modelA_in.add(BatchNormalization())

    modelA_1 = Sequential()
    modelA_2 = Sequential()
    modelA_3 = Sequential()

    modelA_1.add(modelA_in)
    modelA_1.add(Conv2D(num_filters, kernel_size=(filter_sizes[0], EM_DIM), \
            padding='valid', kernel_initializer='normal', activation='relu',\
            kernel_regularizer=L2(beta)))
    modelA_1.add(MaxPool2D(pool_size=(TITLE_LEN - filter_sizes[0] + 1, 1), \
            strides=(1,1), padding='valid'))

    modelA_2.add(modelA_in)
    modelA_2.add(Conv2D(num_filters, kernel_size=(filter_sizes[1], EM_DIM), \
            padding='valid', kernel_initializer='normal', activation='relu',\
            kernel_regularizer=L2(beta)))
    modelA_2.add(MaxPool2D(pool_size=(TITLE_LEN - filter_sizes[1] + 1, 1), \
            strides=(1,1), padding='valid'))

    modelA_3.add(modelA_in)
    modelA_3.add(Conv2D(num_filters, kernel_size=(filter_sizes[2], EM_DIM), \
            padding='valid', kernel_initializer='normal', activation='relu',\
            kernel_regularizer=L2(beta)))
    modelA_3.add(MaxPool2D(pool_size=(TITLE_LEN - filter_sizes[2] + 1, 1), \
            strides=(1,1), padding='valid'))

    modelB_in = Sequential()
    modelB_in.add(Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
            weights = [embedding_mat], input_length = TEXT_LEN, \
            trainable=True))
    modelB_in.add(Reshape((TEXT_LEN, EM_DIM, 1)))

    modelB_in.add(BatchNormalization())

    modelB_1 = Sequential()
    modelB_2 = Sequential()
    modelB_3 = Sequential()

    modelB_1.add(modelB_in)
    modelB_1.add(Conv2D(num_filters, kernel_size=(filter_sizes[0], EM_DIM), \
            padding='valid', kernel_initializer='normal', activation='relu',\
            kernel_regularizer=L2(beta)))
    modelB_1.add(MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[0] + 1, 1), \
            strides=(1,1), padding='valid'))

    modelB_2.add(modelB_in)
    modelB_2.add(Conv2D(num_filters, kernel_size=(filter_sizes[1], EM_DIM), \
            padding='valid', kernel_initializer='normal', activation='relu',\
            kernel_regularizer=L2(beta)))
    modelB_2.add(MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[1] + 1, 1), \
            strides=(1,1), padding='valid'))

    modelB_3.add(modelB_in)
    modelB_3.add(Conv2D(num_filters, kernel_size=(filter_sizes[2], EM_DIM), \
            padding='valid', kernel_initializer='normal', activation='relu',\
            kernel_regularizer=L2(beta)))
    modelB_3.add(MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[2] + 1, 1), \
            strides=(1,1), padding='valid'))

    for i in range(NUM_INPUT):
        maxpoolA_1.append(modelA_1(inputA[i]))
        maxpoolA_2.append(modelA_2(inputA[i]))
        maxpoolA_3.append(modelA_3(inputA[i]))
        maxpoolB_1.append(modelB_1(inputB[i]))
        maxpoolB_2.append(modelB_2(inputB[i]))
        maxpoolB_3.append(modelB_3(inputB[i]))

    ma1 = Add()(maxpoolA_1)
    ma2 = Add()(maxpoolA_2)
    ma3 = Add()(maxpoolA_3)
    mb1 = Add()(maxpoolB_1)
    mb2 = Add()(maxpoolB_2)
    mb3 = Add()(maxpoolB_3)

    concatenated_tensor = Concatenate(axis=3)([ma1, ma2, ma3, mb1, mb2, mb3])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    fc = Dense(hidden_dim1, kernel_regularizer=L2(beta))(dropout)
    bn2 = BatchNormalization()(fc)
    relu = LeakyReLU(0.3)(bn2)
    output = Dense(1, activation='sigmoid', kernel_regularizer=L2(beta))(relu)

    final_model = Model(inputs = inputA+inputB, outputs = output)
    adam = optimizers.Adam(lr=learning_rate)
    final_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    final_model.summary()

    #early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

    #final_model.fit(train_x, train_y, batch_size = bs, epochs = ne)

    #final_model.fit([t for t in train_x], train_y, batch_size = bs, epochs = ne, \
    #        verbose=1, validation_data=([t for t in test_x], test_y))
    final_model.fit(train_x, train_y, batch_size = bs, epochs = ne, \
            verbose=1, validation_data=(test_x, test_y))#, callbacks=[early_stop])

if __name__=='__main__':
    main()
