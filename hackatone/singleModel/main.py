import os 
import pickle
import random
import numpy as np
import tensorflow as tf
from config import get_config
from dataloader import load_text, make_embedding, stock_data, evaluate_data
from pymongo import MongoClient

from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Reshape, Flatten
from keras.layers import LSTM, Dropout, Concatenate, BatchNormalization, LeakyReLU
from keras import optimizers, backend as K
from keras.regularizers import l2 as L2
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session

STOCK_LIST = ['BAC', 'WMT', 'AAPL', 'AMZN', 'MSFT', 'GOOGL', 'FB', 'TSLA', 'NFLX', 'JNJ']

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

def main(config):
    IDX = config.idx
    VOCA_SIZE = config.voca_size
    TITLE_LEN = config.title_len 
    TEXT_LEN = config.text_len
    EM_DIM = config.em_dim

    learning_rate = config.lr
    beta = config.beta
    bs = config.bs
    ne = config.ne
    num_filters = config.num_filters
    hidden_dim1 = config.hidden_dim
    filter_sizes = [3,4,5]

    def model():
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
        out_A1 = model_A1(input_A)
        out_A2 = model_A2(input_A)
        out_A3 = model_A3(input_A)

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
        out_B1 = model_B1(input_B)
        out_B2 = model_B2(input_B)
        out_B3 = model_B3(input_B)

        concatenated_tensor = Concatenate(axis=3)([out_A1, out_A2, out_A3, out_B1, out_B2, out_B3])
        flatten = Flatten()(concatenated_tensor)

        '''
        dropout = Dropout(drop)(flatten)
        fc = Dense(hidden_dim2, kernel_regularizer=L2(beta))(dropout)
        bn = BatchNormalization()(fc)
        flatten = LeakyReLU(0.3)(bn)
        '''

        output = Dense(1, activation='sigmoid', kernel_regularizer=L2(beta))(flatten)
        return Model(inputs = (input_A, input_B), outputs = output)

    if config.evaluate == 0:
        ## preprocessing data ##
        train_dates, train_titles, train_texts = load_text(IDX, True)
        test_dates, test_titles, test_texts = load_text(IDX, False)

        if exist('tokenizer'+str(IDX)):
            tokenizer = load_obj('tokenizer'+str(IDX))
            print('load tokenizer'+str(IDX))
        else:
            tokenizer = Tokenizer(VOCA_SIZE)
            tokenizer.fit_on_texts(train_titles + train_texts + test_titles + test_texts)
            save_obj(tokenizer, 'tokenizer'+str(IDX))
            print('save tokenizer'+str(IDX))

        if exist('embedding_mat'+str(IDX)):
            embedding_mat = load_obj('embedding_mat'+str(IDX))
        else:
            embedding_mat = make_embedding(tokenizer, VOCA_SIZE, EM_DIM)
            save_obj(embedding_mat, 'embedding_mat'+str(IDX))

        if exist('total_data'+str(IDX)):
            print('import total data')
            seq_train_1, seq_train_2, train_labels, seq_test_1, seq_test_2, test_labels = load_obj('total_data'+str(IDX))
        else:
            seq_train_1 = pad_sequences(tokenizer.texts_to_sequences(train_titles), maxlen=TITLE_LEN)
            seq_train_2 = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=TEXT_LEN)
            seq_test_1 = pad_sequences(tokenizer.texts_to_sequences(test_titles), maxlen=TITLE_LEN)
            seq_test_2 = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=TEXT_LEN)

            stock, sdate, updown = zip(*stock_data(IDX))
            date2updown = dict(zip(sdate, updown))
            train_labels = list(map(lambda d: date2updown[d], train_dates))
            test_labels = list(map(lambda d: date2updown[d], test_dates))

            total_data = [seq_train_1, seq_train_2, train_labels, seq_test_1, seq_test_2, test_labels]
            save_obj(total_data, 'total_data'+str(IDX))

        n_symbols = len(embedding_mat)
        print(n_symbols)

        print('total:', len(train_labels)+len(test_labels))
        train_zip = list(zip(seq_train_1, seq_train_2, train_labels))
        random.shuffle(train_zip)
        seq_train_1, seq_train_2, train_labels = zip(*train_zip)

        train_x1 = np.array(seq_train_1)
        train_x2 = np.array(seq_train_2)
        train_y = np.array(train_labels)
        test_x1 = np.array(seq_test_1)
        test_x2 = np.array(seq_test_2)
        test_y = np.array(test_labels)

        ## model ##
        K.clear_session()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess=tf.Session(config=config)
        set_session(sess)

        print(embedding_mat.shape)

        final_model = model()
        adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
        final_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        final_model.summary()

        filepath="ckpt/model-%d-{epoch:02d}-{val_acc:.2f}.hdf5"%IDX
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

        final_model.fit([train_x1, train_x2], train_y, batch_size = bs, epochs = ne, \
                verbose=1, validation_data=([test_x1, test_x2], test_y), callbacks=[checkpoint])

    elif config.evaluate == 1:
        input_x = evaluate_data(IDX)
        tokenizer = load_obj('tokenizer'+str(IDX))
        embedding_mat = load_obj('embedding_mat'+str(IDX))
        n_symbols = len(embedding_mat)

        seq_1 = pad_sequences(tokenizer.texts_to_sequences(input_x[0]), maxlen=TITLE_LEN)
        seq_2 = pad_sequences(tokenizer.texts_to_sequences(input_x[1]), maxlen=TEXT_LEN)

        final_model = model()
        final_model.load_weights('ckpt/good-%d.hdf5'%IDX)

        predict = final_model.predict([seq_1, seq_2])
        print(predict[50:80])
        print(sum(predict))

        assert len(predict)==len(input_x[0])
        for i in range(len(predict)):
            db.news.update({'title':input_x[0][i], 'text':input_x[1][i]}, \
                    {'$set': {'predict':float(predict[i][0])}})

if __name__ == '__main__':
    config = get_config()
    main(config)
