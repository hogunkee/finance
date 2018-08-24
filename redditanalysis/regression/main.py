import numpy as np
import pandas as pd
import tensorflow as tf
from config import get_config
from data_loader import load_data
from model import *
from w2v import fasttext

from keras import optimizers, backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

## utility functions ##
def splitt(title):
    return title.split()

def embed(split_title):
    return w2vmodel[split_title]

def padding(embed_vec):
    pad_len = TITLE_LEN - len(embed_vec)
    return np.pad(embed_vec, ((pad_len, 0), (0, 0)), 'constant')

def embed_pad(split_title):
    global TITLE_LEN
    global w2vmodel
    pad_len = TITLE_LEN - len(split_title)
    return np.pad(w2vmodel[split_title], ((pad_len, 0), (0, 0)), 'constant')

def generator(df, batch_size, title_len, em_dim):
    df = df.sample(frac=1).reset_index(drop=True)
    while True:
        for i in range(len(df)//batch_size):
            x = df.loc[batch_size*i:batch_size*(i+1)-1, 'split_title'].apply(embed_pad)
            y = df.loc[batch_size*i:batch_size*(i+1)-1, 'log_score']
            x_gen = np.array(list(x))
            y_gen = list(y)
            if x_gen.shape == (batch_size, title_len, em_dim):
                yield x_gen, y_gen

class PrecisionEvaluation(Callback):
    def __init__(self, validation_df, interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.x = np.array(list(validation_df['split_title'].apply(embed_pad)))
        self.y = validation_df['log_score']
        self.date = validation_df['date']
        self.max = 0.0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x)
            y_pred = np.concatenate(y_pred)
            precision = self.evaluate_by_top_score(self.date, self.y, y_pred)
            # save max precision
            if precision > self.max:
                self.max = precision
            print("Evaluation - epoch: {:d} - precision: [{:.6f}]".format(epoch, precision))
            print('max precision: %.6f' %self.max)
            print()

    @staticmethod
    def evaluate_by_top_score(date, score, pred):
        df = pd.DataFrame({'date':date, 'score':score, 'pred':pred})
        date_list = pd.unique(date)

        df['score_label'] = np.zeros(len(df), dtype=np.int32)
        df['pred_label'] = np.zeros(len(df), dtype=np.int32)
        top_score = []
        top_pred = []

        for day in date_list:
            df_by_date = df[df['date'] == day]
            topk = int(0.1 * len(df_by_date))
            top_score.extend(df_by_date.score.nlargest(topk).index)
            top_pred.extend(df_by_date.pred.nlargest(topk).index)

        df['score_label'][top_score] = 1
        df['pred_label'][top_pred] = 1
        cross = pd.crosstab(df['score_label'], df['pred_label'], rownames=["Actual"], \
                colnames=["Predicted"])
        precision = cross[1][1] / (cross[1][1] + cross[0][1])
        print(cross)

        return precision

def main(config):
    global TITLE_LEN
    TITLE_LEN = config.title_len 
    EM_DIM = config.em_dim

    learning_rate = config.lr
    beta = config.beta
    bs = config.bs
    ne = config.ne

    start_date = config.start_date
    valid_date = config.valid_date

    ## preprocessing data ##
    total_data = load_data()
    global w2vmodel
    w2vmodel = fasttext()

    total_data['split_title'] = total_data['title'].apply(splitt)
    #total_data['embed'] = total_data['title'].apply(splitt).apply(embed).apply(padding)

    if start_date == '0000-00-00':
        train_df = total_data[total_data.date.apply(str)<'2018-05-01']
    else:
        train_df = total_data[(total_data.date.apply(str)<valid_date) \
                & (total_data.date.apply(str)>=start_date)]

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    valid_df = total_data[total_data.date.apply(str)>=valid_date]

    print('train data:', len(train_df))
    print('test data:', len(valid_df))

    ## model ##
    K.clear_session()
    tfconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess=tf.Session(config=tfconfig)
    set_session(sess)

    model = cnnmodel(config)
    final_model = model()
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    final_model.compile(loss='mse', optimizer=adam)
    final_model.summary()

    pval = PrecisionEvaluation(valid_df, 1)

    #filepath="ckpt/model-%d-{epoch:02d}-{val_acc:.2f}.hdf5"%IDX
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

    print('Training Start!!!')
    final_model.fit_generator(generator(train_df, bs, TITLE_LEN, EM_DIM), \
            steps_per_epoch=len(train_df)//bs, nb_epoch=ne, callbacks = [pval])

    print('Max Precision:', pval.max)

if __name__ == '__main__':
    config = get_config()
    main(config)
