import os 
import pickle
import random
import numpy as np
import tensorflow as tf
from dataloader import load_data, make_embedding, EM_DIM

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

CSV_DIR = '../data/Full_Data.csv'
VOCA_SIZE = 70000
TEXT_LEN = 925

learning_rate = 1e-2
beta = 1e-5
hidden_dim1 = 150
hidden_dim2 = 200
filter_sizes = [3, 4, 5]
num_filters = 64
drop = 0.5
bs = 126
ne = 100

def save_obj(obj, name):
    with open('obj/'+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)

def exist(name):
    return os.path.exists('obj/'+name+'.pkl')

def main():
    train, test = load_data(CSV_DIR)
    tmp = list(zip(train[0], train[1]))
    random.shuffle(tmp)
    x,y = zip(*tmp)
    train = [x,y]
    train_label = np.asarray(train[1], dtype='int32')
    test_label = np.asarray(test[1], dtype='int32')
    train = [train[0], train_label]
    test = [test[0], test_label]


    if exist('tokenizer'):
        tokenizer = load_obj('tokenizer')
        print('load tokenizer')
    else:
        tokenizer = Tokenizer(VOCA_SIZE)
        tokenizer.fit_on_texts(train[0])
        save_obj(tokenizer, 'tokenizer')
        print('save tokenizer')

    if exist('embedding_mat'):
        embedding_mat = load_obj('embedding_mat')
    else:
        embedding_mat = make_embedding(tokenizer, VOCA_SIZE)
        save_obj(embedding_mat, 'embedding_mat')

    train_seq = pad_sequences(tokenizer.texts_to_sequences(train[0]), maxlen=TEXT_LEN)
    test_seq = pad_sequences(tokenizer.texts_to_sequences(test[0]), maxlen=TEXT_LEN)

    K.clear_session()
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    #config.log_device_placement = True
    sess=tf.Session(config=config)
    set_session(sess)

    n_symbols = len(embedding_mat)
    print(embedding_mat.shape)
    model = Sequential()
    model.add(Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
            weights = [embedding_mat], input_length = TEXT_LEN, \
            trainable=True))
    model.add(BatchNormalization())
    model.add(LSTM(hidden_dim1, return_sequences=True))
    #model.add(BatchNormalization())
    model.add(Reshape((TEXT_LEN, hidden_dim1, 1)))
    
    model1 = Sequential()
    model2 = Sequential()
    model3 = Sequential()
    model1.add(model)
    model2.add(model)
    model3.add(model)

    model1.add(Conv2D(num_filters, kernel_size=(filter_sizes[0], hidden_dim1), \
            padding='valid', kernel_initializer='normal',\
            kernel_regularizer=L2(beta)))
    model2.add(Conv2D(num_filters, kernel_size=(filter_sizes[1], hidden_dim1), \
            padding='valid', kernel_initializer='normal',\
            kernel_regularizer=L2(beta)))
    model3.add(Conv2D(num_filters, kernel_size=(filter_sizes[2], hidden_dim1), \
            padding='valid', kernel_initializer='normal',\
            kernel_regularizer=L2(beta)))

    model1.add(BatchNormalization())
    model2.add(BatchNormalization())
    model3.add(BatchNormalization())

    model1.add(LeakyReLU(0.3))
    model2.add(LeakyReLU(0.3))
    model3.add(LeakyReLU(0.3))

    model1.add(MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[0] + 1, 1), \
            strides=(1,1), padding='valid'))
    model2.add(MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[1] + 1, 1), \
            strides=(1,1), padding='valid'))
    model3.add(MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[2] + 1, 1), \
            strides=(1,1), padding='valid'))

    input_a = Input(shape=(TEXT_LEN,))
    out1 = model1(input_a)
    out2 = model2(input_a)
    out3 = model3(input_a)

    concatenated_tensor = Concatenate(axis=3)([out1, out2, out3])
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
