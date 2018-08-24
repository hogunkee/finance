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
from keras.layers import Dropout, Concatenate, BatchNormalization, LeakyReLU
from keras import optimizers, backend as K
from keras.callbacks import EarlyStopping, Callback
from keras.backend.tensorflow_backend import set_session

CSV_DIR = '../data/Full_Data.csv'
VOCA_SIZE = 70000
TEXT_LEN = 925

learning_rate = 1e-3
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5
bs = 300
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
    print(type(test[1]))
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

    ## model ##
    input_a = Input(shape=(TEXT_LEN,))
    embedding = Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
            weights = [embedding_mat], input_length = TEXT_LEN, \
            trainable=True)(input_a)
    reshape = Reshape((TEXT_LEN, EM_DIM, 1))(embedding)

    bn = BatchNormalization()(reshape)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], EM_DIM), \
            padding='valid', kernel_initializer='normal', activation='relu')(bn)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], EM_DIM), \
            padding='valid', kernel_initializer='normal', activation='relu')(bn)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], EM_DIM), \
            padding='valid', kernel_initializer='normal', activation='relu')(bn)

    maxpool_0 = MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[0] + 1, 1), \
            strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[1] + 1, 1), \
            strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(TEXT_LEN - filter_sizes[2] + 1, 1), \
            strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    fc = Dense(200)(dropout)
    #fc = Dense(200, activation='relu')(dropout)
    bn2 = BatchNormalization()(fc)
    relu = LeakyReLU(bn2)
    output = Dense(1, activation='sigmoid')(relu)

    final_model = Model(inputs = input_a, outputs = output)
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    final_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    final_model.summary()

    #early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

    final_model.fit(train_seq, train[1], batch_size = bs, epochs = ne, \
            verbose=1, validation_data=(test_seq, test[1]))#, callbacks=[early_stop])
    '''
    final_model.fit(train_seq, train[1], batch_size = bs, epochs = ne, \
            verbose=1, callbacks=[TestCallback((test_seq, test[1]))])
    '''

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        predict = self.model.predict(x)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        #print(predict)

if __name__ == '__main__':
    main()
