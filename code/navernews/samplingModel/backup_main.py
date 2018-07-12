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
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session

VOCA_SIZE = 70000
EM_DIM = 300
TEXT_DIR = '../../data/news/'
STOCK_DIR = '../../data/stock.csv'
VALIDATION = 0.1
TITLE_LEN = 15
TEXT_LEN = 300
NUM_INPUT = 10
GEN_RANDOM = 10
#GEN_RANDOM = 100

test_sample = 90
learning_rate = 0.1
hidden_dim1 = 200
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5
bs = 10
ne = 5

def save_obj(obj, name):
    with open('obj/'+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)

def exist(name):
    return os.path.exists('obj/'+name+'.pkl')

def sampling(data):
    sampled_titles = []
    sampled_texts = []
    sampled_labels = []
    titles, texts, labels = zip(*data)
    for x in range(len(titles)):
        title = titles[x]
        text = texts[x]
        label = labels[x]
        for y in range(GEN_RANDOM):
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

train_data = sampling(train_data)
test_data = sampling(test_data)
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

    input_a = [Input(shape=(TITLE_LEN,)) for k in range(NUM_INPUT)]
    input_b = [Input(shape=(TEXT_LEN,)) for k in range(NUM_INPUT)]
    embed_a, reshape_a, bn_a, conv_a0, conv_a1, conv_a2, maxpool_a0, maxpool_a1, maxpool_a2,\
        embed_b, reshape_b, bn_b, conv_b0, conv_b1, conv_b2, maxpool_b0, maxpool_b1, maxpool_b2\
        = [[] for _ in range(18)]

    for i in range(NUM_INPUT):
        embed_a.append(Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
                weights = [embedding_mat], input_length = TITLE_LEN, \
                trainable=True)(input_a[i]))

        reshape_a.append(Reshape((TITLE_LEN, EM_DIM, 1))(embed_a[i]))
        bn_a.append(BatchNormalization()(reshape_a[i]))

        conv_a0.append(Conv2D(num_filters, kernel_size=(filter_sizes[0], EM_DIM), \
                padding='valid', kernel_initializer='normal', activation='relu')(bn_a[i]))
        conv_a1.append(Conv2D(num_filters, kernel_size=(filter_sizes[1], EM_DIM), \
                padding='valid', kernel_initializer='normal', activation='relu')(bn_a[i]))
        conv_a2.append(Conv2D(num_filters, kernel_size=(filter_sizes[2], EM_DIM), \
                padding='valid', kernel_initializer='normal', activation='relu')(bn_a[i]))

        maxpool_a0.append(MaxPool2D(pool_size=(TITLE_LEN - filter_sizes[0] + 1, 1), \
                strides=(1,1), padding='valid')(conv_a0[i]))
        maxpool_a1.append(MaxPool2D(pool_size=(TITLE_LEN - filter_sizes[1] + 1, 1), \
                strides=(1,1), padding='valid')(conv_a1[i]))
        maxpool_a2.append(MaxPool2D(pool_size=(TITLE_LEN - filter_sizes[2] + 1, 1), \
                strides=(1,1), padding='valid')(conv_a2[i]))

        embed_b.append(Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
                weights = [embedding_mat], input_length = TEXT_LEN, \
                trainable=True)(input_b[i]))
        reshape_b.append(Reshape((TEXT_LEN, EM_DIM, 1))(embed_b[i]))
        bn_b.append(BatchNormalization()(reshape_b[i]))

        conv_b0.append(Conv2D(num_filters, kernel_size=(filter_sizes[0], EM_DIM), \
                padding='valid', kernel_initializer='normal', activation='relu')(bn_b[i]))
        conv_b1.append(Conv2D(num_filters, kernel_size=(filter_sizes[1], EM_DIM), \
                padding='valid', kernel_initializer='normal', activation='relu')(bn_b[i]))
        conv_b2.append(Conv2D(num_filters, kernel_size=(filter_sizes[2], EM_DIM), \
                padding='valid', kernel_initializer='normal', activation='relu')(bn_b[i]))

        maxpool_b0.append(MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[0] + 1, 1), \
                strides=(1,1), padding='valid')(conv_b0[i]))
        maxpool_b1.append(MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[1] + 1, 1), \
                strides=(1,1), padding='valid')(conv_b1[i]))
        maxpool_b2.append(MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[2] + 1, 1), \
                strides=(1,1), padding='valid')(conv_b2[i]))

    ma0 = Add()(maxpool_a0)
    ma1 = Add()(maxpool_a1)
    ma2 = Add()(maxpool_a2)
    mb0 = Add()(maxpool_b0)
    mb1 = Add()(maxpool_b1)
    mb2 = Add()(maxpool_b2)

    concatenated_tensor = Concatenate(axis=1)([ma0, ma1, ma2, mb0, mb1, mb2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    fc = Dense(hidden_dim1)(dropout)
    bn2 = BatchNormalization()(fc)
    relu = LeakyReLU(0.3)(bn2)
    output = Dense(1, activation='sigmoid')(relu)

    final_model = Model(inputs = input_a+input_b, outputs = output)
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
