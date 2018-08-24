import datetime
import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
import tensorflow_hub as hub
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity

tf.logging.set_verbosity(0)
start_date = '0000-00-00'
#start_date = '2018-04-01'
valid_date = '2018-05-01'

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def labeling_data(ratio):
    print('labeling data')
    df = load_obj('total_data')
    date_list = pd.unique(df['date'])

    df['label'] = np.zeros(len(df), dtype=np.int32)
    top_score = []

    for day in date_list:
        df_by_date = df[df['date'] == day]
        topk = int(ratio * len(df_by_date))
        top_score.extend(df_by_date.score.nlargest(topk).index)

    df['label'][top_score] = 1

    save_obj(df, 'total_data')
    del df

#labeling_data(0.1)

total_data = load_obj('total_data')
print('total data:', len(total_data))

if start_date == '0000-00-00':
    train_df = total_data[total_data.date.apply(str)<'2018-05-01']
else:
    train_df = total_data[(total_data.date.apply(str)<valid_date) \
            & (total_data.date.apply(str)>=start_date)]

train_df = train_df.sample(frac=1).reset_index(drop=True)
valid_df = total_data[total_data.date.apply(str)>=valid_date]

print('train data:', len(train_df))
print('test data:', len(valid_df))

del total_data

class fcmodel(object):
    def __init__(self):
        self.bs = 10000
        self.em_dim = 512
        self.beta = 1e-4
        self.fc_dim = 256
        self.fc_dim2 = 64
        self.fc_dim3 = 16
        self.lr = 1e-3
        self.batch_x = tf.placeholder(tf.float32, [None, self.em_dim])
        self.batch_y = tf.placeholder(tf.float32, [None, 1])
        self.create_model()

    def create_model(self):
        # size : (bs, 512)
        batch_x = self.batch_x
        batch_y = self.batch_y
        with tf.device("/gpu:0"):
            with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
                W1 = tf.get_variable('weight-1', [self.em_dim, self.fc_dim], dtype=tf.float32)
                b1 = tf.get_variable('bias-1', [self.fc_dim], dtype=tf.float32)
                W2 = tf.get_variable('weight-2', [self.fc_dim, self.fc_dim2], dtype=tf.float32)
                b2 = tf.get_variable('bias-2', [self.fc_dim2], dtype=tf.float32)
                W3 = tf.get_variable('weight-3', [self.fc_dim2, self.fc_dim3], dtype=tf.float32)
                b3 = tf.get_variable('bias-3', [self.fc_dim3], dtype=tf.float32)
                W4 = tf.get_variable('weight-4', [self.fc_dim3, 1], dtype=tf.float32)
                b4 = tf.get_variable('bias-4', [1], dtype=tf.float32)

        print('x shape:', batch_x.shape)
        h = tf.matmul(batch_x, W1) + b1
        h = tf.nn.tanh(h)
        h = tf.matmul(h, W2) + b2
        h = tf.nn.tanh(h)
        h = tf.matmul(h, W3) + b3
        h = tf.nn.tanh(h)
        h = tf.matmul(h, W4) + b4
        y = tf.nn.sigmoid(h)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_y, logits=h))
        regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
        self.regul_loss = self.beta * regularizer
        self.loss = loss + self.regul_loss

        with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.train_op = train_op #= tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        self.predict = y
        correct_prediction = tf.equal(tf.round(self.predict), batch_y)
        self.accur = accur = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.variable_scope('embed', reuse=tf.AUTO_REUSE):
    embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/1')

def embedding(sentence_list):
    with tf.device("/cpu:0"):
        with tf.variable_scope('embed', reuse=tf.AUTO_REUSE):
            return embed(sentence_list)

fraction = 0.5
sess = tf.Session(config=tf.ConfigProto(
        #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=fraction),
        gpu_options=tf.GPUOptions(allow_growth=True),
        #log_device_placement=True))
        ))

model = fcmodel()
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

def evaluate_by_top_score(df, pred, ratio_pred=0.1):
    date_list = pd.unique(list(df['date']))
    df['pred'] = np.zeros(len(df), dtype=np.float32)
    df['pred'] = pred

    df['pred_label'] = np.zeros(len(df), dtype=np.int32)
    top_pred = []

    for day in date_list:
        df_by_date = df[df['date'] == day]
        topk_pred = int(ratio_pred * len(df_by_date))
        top_pred.extend(df_by_date.pred.nlargest(topk_pred).index)

    df['pred_label'][top_pred] = 1
    cross = pd.crosstab(df['label'], df['pred_label'], rownames=["Actual"], \
            colnames=["Predicted"])
    recall = cross[1][1] / (cross[1][1] + cross[0][1])
    print(cross)
    return recall


ne = 10
num_steps = 10
bs = model.bs # 100

df_0 = train_df[train_df['label']==0]
df_1 = train_df[train_df['label']==1]
del train_df

#training
for epoch in range(ne):
    avg_accur = 0.0
    print('Epoch [%d/%d]' %(epoch+1, ne))
    for step in range(num_steps):
        batch_df_0 = df_0.sample(n=int(bs/2), replace=True)
        batch_df_1 = df_1.sample(n=int(bs/2), replace=True)
        batch_df = pd.concat([batch_df_0, batch_df_1])
        del batch_df_0, batch_df_1

        x = sess.run(embedding(list(batch_df['title'])))
        y = np.array(list(batch_df['label'])).reshape([-1,1])
        del batch_df

        loss, accur,_, pred = sess.run([model.loss, model.accur, model.train_op, model.predict],\
                                        feed_dict={model.batch_x: x, model.batch_y: y})
        print('    Steps [%d/%d] - loss: %.4f / accur: %.4f' \
                %(step+1, num_steps, loss, accur))
        #print(pred)
        del x, y
        avg_accur += accur

        '''
        for v in tf.get_default_graph().as_graph_def().node:
            print(v.name)
        '''

    avg_accur /= num_steps
    print('Training accuracy:', avg_accur)

    x = sess.run(embedding(list(valid_df['title'])))
    #y = valid_df['label']
    y = np.array(list(valid_df['label'])).reshape([-1,1])
    pred = sess.run(model.predict, feed_dict={model.batch_x: x})
    recall = evaluate_by_top_score(valid_df, pred)
    print('recall:', recall)
    del x, y

