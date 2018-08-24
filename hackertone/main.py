import os 
import pickle
import random
import numpy as np
import tensorflow as tf
from dataloader import load_data, make_embedding, EM_DIM
from pymongo import MongoClient

from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Reshape, Flatten
from keras.layers import LSTM, Dropout, Concatenate, BatchNormalization, LeakyReLU
from keras import optimizers, backend as K
from keras.regularizers import l2 as L2
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session

VOCA_SIZE = 70000
TITLE_LEN = 20
TEXT_LEN = 300
EM_DIM = 100

learning_rate = 1e-2
beta = 1e-5
hidden_dim1 = 150
hidden_dim2 = 200
filter_sizes = [3, 4, 5]
num_filters = 64
drop = 0.5
bs = 126
ne = 100

STOCK_LIST = ['BAC', 'WMT', 'AAPL', 'AMZN', 'MSFT', 'GOOGL', 'FB', 'TSLA', 'NFLX', 'JNJ']
STOCK_IDX = 1

# db load #
client = MongoClient('localhost', 27017)
db = client['hackathon']

rows = db.news.find()

def save_obj(obj, name):
    with open('obj/'+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)

def exist(name):
    return os.path.exists('obj/'+name+'.pkl')

def main():
    if exist('tokenizer'):
        tokenizer = load_obj('tokenizer')
        print('load tokenizer')
    else:
        tokenizer = Tokenizer(VOCA_SIZE)
        dates, titles, texts = load_text() #TODO
        tokenizer.fit_on_texts(titles + texts)
        save_obj(tokenizer, 'tokenizer')
        print('save tokenizer')

    if exist('embedding_mat'):
        embedding_mat = load_obj('embedding_mat')
    else:
        embedding_mat = make_embedding(tokenizer, VOCA_SIZE, EM_DIM)
        save_obj(embedding_mat, 'embedding_mat')

    if exist('total_data'):
        print('import total data')
        total_data = load_obj('total_data')
    else:
        titles_seq = []
        texts_seq = []
        for t in titles:
            t_seq = pad_sequences(tokenizer.texts_to_sequences(t), maxlen=TITLE_LEN)
            titles_seq.append(t_seq)
        for t in texts:
            t_seq = pad_sequences(tokenizer.texts_to_sequences(t), maxlen=TEXT_LEN)
            texts_seq.append(t_seq)

        stock, sdate, updown = zip(*stock_data(STOCK_LIST))
        date2updown = dict(zip(sdate, updown))
        labels = list(map(lambda d: date2updown[d], dates))
        total_data = np.array(list(zip(titles_seq, texts_seq, labels)))
        save_obj(total_data, 'total_data')


    n_symbols = len(embedding_mat)
    print(n_symbols)

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


    ## model ##
    K.clear_session()
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess=tf.Session(config=config)
    set_session(sess)

    print(embedding_mat.shape)

    ## A model ##
    model_A = Sequential()
    model_A.add(Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
            weights = [embedding_mat], input_length = TITLE_LEN, \
            trainable=True))
    model_A.add(BatchNormalization())
    model_A.add(LSTM(hidden_dim1, return_sequences=True))
    #model_A.add(BatchNormalization())
    model_A.add(Reshape((TITLE_LEN, hidden_dim1, 1)))
    
    model_A1 = Sequential()
    model_A2 = Sequential()
    model_A3 = Sequential()
    model_A1.add(model_A)
    model_A2.add(model_A)
    model_A3.add(model_A)

    model_A1.add(Conv2D(num_filters, kernel_size=(filter_sizes[0], hidden_dim1), \
            padding='valid', kernel_initializer='normal',\
            kernel_regularizer=L2(beta)))
    model_A2.add(Conv2D(num_filters, kernel_size=(filter_sizes[1], hidden_dim1), \
            padding='valid', kernel_initializer='normal',\
            kernel_regularizer=L2(beta)))
    model_A3.add(Conv2D(num_filters, kernel_size=(filter_sizes[2], hidden_dim1), \
            padding='valid', kernel_initializer='normal',\
            kernel_regularizer=L2(beta)))

    model_A1.add(BatchNormalization())
    model_A2.add(BatchNormalization())
    model_A3.add(BatchNormalization())

    model_A1.add(LeakyReLU(0.3))
    model_A2.add(LeakyReLU(0.3))
    model_A3.add(LeakyReLU(0.3))

    model_A1.add(MaxPool2D(pool_size=(TITLE_LEN - filter_sizes[0] + 1, 1), \
            strides=(1,1), padding='valid'))
    model_A2.add(MaxPool2D(pool_size=(TITLE_LEN - filter_sizes[1] + 1, 1), \
            strides=(1,1), padding='valid'))
    model_A3.add(MaxPool2D(pool_size=(TITLE_LEN - filter_sizes[2] + 1, 1), \
            strides=(1,1), padding='valid'))

    input_A = Input(shape=(TITLE_LEN,))
    out_A1 = model1(input_A)
    out_A2 = model2(input_A)
    out_A3 = model3(input_A)

    ## B model ##
    model_B = Sequential()
    model_B.add(Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
            weights = [embedding_mat], input_length = TEXT_LEN, \
            trainable=True))
    model_B.add(BatchNormalization())
    model_B.add(LSTM(hidden_dim1, return_sequences=True))
    #model_A.add(BatchNormalization())
    model_B.add(Reshape((TEXT_LEN, hidden_dim1, 1)))
    
    model_B1 = Sequential()
    model_B2 = Sequential()
    model_B3 = Sequential()
    model_B1.add(model_B)
    model_B2.add(model_B)
    model_B3.add(model_B)

    model_B1.add(Conv2D(num_filters, kernel_size=(filter_sizes[0], hidden_dim1), \
            padding='valid', kernel_initializer='normal',\
            kernel_regularizer=L2(beta)))
    model_B2.add(Conv2D(num_filters, kernel_size=(filter_sizes[1], hidden_dim1), \
            padding='valid', kernel_initializer='normal',\
            kernel_regularizer=L2(beta)))
    model_B3.add(Conv2D(num_filters, kernel_size=(filter_sizes[2], hidden_dim1), \
            padding='valid', kernel_initializer='normal',\
            kernel_regularizer=L2(beta)))

    model_B1.add(BatchNormalization())
    model_B2.add(BatchNormalization())
    model_B3.add(BatchNormalization())

    model_B1.add(LeakyReLU(0.3))
    model_B2.add(LeakyReLU(0.3))
    model_B3.add(LeakyReLU(0.3))

    model_B1.add(MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[0] + 1, 1), \
            strides=(1,1), padding='valid'))
    model_B2.add(MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[1] + 1, 1), \
            strides=(1,1), padding='valid'))
    model_B3.add(MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[2] + 1, 1), \
            strides=(1,1), padding='valid'))

    input_B = Input(shape=(TEXT_LEN,))
    out_B1 = model1(input_B)
    out_B2 = model2(input_B)
    out_B3 = model3(input_B)

    concatenated_tensor = Concatenate(axis=3)([out_A1, out_A2, out_A3, out_B1, out_B2, out_B3])
    flatten = Flatten()(concatenated_tensor)

    '''
    dropout = Dropout(drop)(flatten)
    fc = Dense(hidden_dim2, kernel_regularizer=L2(beta))(dropout)
    bn = BatchNormalization()(fc)
    flatten = LeakyReLU(0.3)(bn)
    '''

    output = Dense(1, activation='sigmoid', kernel_regularizer=L2(beta))(flatten)

    final_model = Model(inputs = input_a, outputs = output)
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    final_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    final_model.summary()

    #early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

    final_model.fit(train_seq, train[1], batch_size = bs, epochs = ne, \
            verbose=1, validation_data=(test_seq, test[1]))#, callbacks=[early_stop])

if __name__ == '__main__':
    main()
