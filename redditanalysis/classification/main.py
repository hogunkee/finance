import numpy as np
import pandas as pd
import tensorflow as tf
from config import get_config
from data_loader import load_data
from model import *
from nltk.corpus import stopwords
from stop_words import get_stop_words
from w2v import fasttext

from keras import optimizers, backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

## utility functions ##
def splitt(title):
    title_split = title.split()
    title_clean = [w for w in title_split if not w in stop_words]
    return title_clean

def embed(split_title):
    return w2vmodel[split_title]

def padding(embed_vec):
    pad_len = TITLE_LEN - len(embed_vec)
    return np.pad(embed_vec, ((pad_len, 0), (0, 0)), 'constant')

def embed_pad(split_title):
    #global TITLE_LEN
    #global w2vmodel
    pad_len = TITLE_LEN - len(split_title)
    return np.pad(w2vmodel[split_title], ((pad_len, 0), (0, 0)), 'constant')

def generator(df, batch_size, title_len, em_dim):
    df_0 = df[df['label']==0]
    df_1 = df[df['label']==1]
    while True:
        for i in range(len(df)//batch_size):
            batch_df_0 = df_0.sample(n=batch_size//2, replace=True)
            batch_df_1 = df_1.sample(n=batch_size - batch_size//2, replace=True)
            batch_df = pd.concat([batch_df_0, batch_df_1])
            x = batch_df['split_title'].apply(embed_pad)
            y = batch_df['label']
            x_gen = np.array(list(x))
            y_gen = list(y)
            if x_gen.shape == (batch_size, title_len, em_dim):
                yield x_gen, y_gen

class RecallEvaluation(Callback):
    def __init__(self, validation_df, interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.x = np.array(list(validation_df['split_title'].apply(embed_pad)))
        self.y = validation_df['label']
        self.date = validation_df['date']
        self.recall_max = 0.0
        self.precision_max = 0.0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x)
            y_pred = np.concatenate(y_pred)
            recall, precision = self.evaluate(self.y, y_pred)
            recall_top1 = self.evaluate_by_top_score(self.date, self.y, y_pred)
            print('top 10%% precision: %.6f' %recall_top1)

            # save max 
            if recall+precision > self.recall_max + self.precision_max:
            #if recall > self.recall_max and precision > self.precision_max:
                self.recall_max = recall 
                self.precision_max = precision

            print("Evaluation - epoch: %d - recall: [%.6f] / precision: [%.6f]"\
                    %(epoch, recall, precision))
            print('max recall: %.6f / precision: %.6f' %(self.recall_max, self.precision_max))
            print()

    @staticmethod
    def evaluate(label, pred):
        assert len(label) == len(pred)
        df = pd.DataFrame({'label':label, 'pred':pred})

        df['pred'] = np.array(pred)
        df['pred_label'] = np.zeros(len(df), dtype=np.int32)
        df['pred_label'][df['pred'] > 0.5] = 1

        cross = pd.crosstab(df['label'], df['pred_label'], rownames=["Actual"], \
                colnames=["Predicted"])
        recall = cross[1][1] / (cross[1][1] + cross[0][1])
        precision = cross[1][1] / (cross[1][1] + cross[1][0])
        #print(cross)

        return recall, precision

    @staticmethod
    def evaluate_by_top_score(date, score, pred, ratio_score=0.1, ratio_pred=0.1):
        df = pd.DataFrame({'date':date, 'score':score, 'pred':pred})
        date_list = pd.unique(date)

        df['score_label'] = np.zeros(len(df), dtype=np.int32)
        df['pred_label'] = np.zeros(len(df), dtype=np.int32)
        top_score = []
        top_pred = []

        for day in date_list:
            df_by_date = df[df['date'] == day]
            topk_score = int(ratio_score * len(df_by_date))
            topk_pred = int(ratio_pred * len(df_by_date))
            top_score.extend(df_by_date.score.nlargest(topk_score).index)
            top_pred.extend(df_by_date.pred.nlargest(topk_pred).index)

        df['score_label'][top_score] = 1
        df['pred_label'][top_pred] = 1
        cross = pd.crosstab(df['score_label'], df['pred_label'], rownames=["Actual"], \
                colnames=["Predicted"])
        recall = cross[1][1] / (cross[1][1] + cross[0][1])
        print(cross)

        return recall 

def main(config):
    global TITLE_LEN
    TITLE_LEN = config.title_len 
    EM_DIM = config.em_dim

    learning_rate = config.lr
    beta = config.beta
    bs = config.bs
    ne = config.ne
    num_steps = config.num_steps

    start_date = config.start_date
    valid_date = config.valid_date

    ## preprocessing data ##
    total_data = load_data()
    global w2vmodel
    w2vmodel = fasttext()

    ## stop words ##
    global stop_words
    stop_words = list(get_stop_words('en'))
    nltk_words = list(stopwords.words('english'))
    stop_words.extend(nltk_words)

    total_data['split_title'] = total_data['title'].apply(splitt)
    #total_data['embed'] = total_data['title'].apply(splitt).apply(embed).apply(padding)

    total_data = total_data[total_data['split_title'].apply(len) > 5]
    print('max length ::::', max(total_data['split_title'].apply(len)))

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

    if config.model == 1:
        model = cnnmodel(config)
    elif config.model == 2:
        model = cnnmodel2(config)
    elif config.model == 3:
        model = cnnmodel3(config)
    elif config.model == 4:
        model = rnnmodel(config)
    elif config.model == 5:
        model = cnnrnnmodel(config)
    else:
        print('Choose correct model!')
        exit()

    final_model = model()
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    final_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    final_model.summary()

    rval = RecallEvaluation(valid_df, 1)

    #filepath="ckpt/model-%d-{epoch:02d}-{val_acc:.2f}.hdf5"%IDX
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

    print('Training Start!!!')
    final_model.fit_generator(generator(train_df, bs, TITLE_LEN, EM_DIM), \
            steps_per_epoch=num_steps, nb_epoch=ne, callbacks=[rval])
    #steps_per_epoch=len(train_df)//bs, nb_epoch=ne, callbacks=[rval])

    print('Max Recall:', rval.max)

if __name__ == '__main__':
    config = get_config()
    main(config)
