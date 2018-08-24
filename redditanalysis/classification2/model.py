from keras.layers.embeddings import Embedding
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Reshape, Flatten
from keras.layers import LSTM, Dropout, Concatenate, BatchNormalization, LeakyReLU
from keras.models import Sequential, Model
from keras.regularizers import l2 as L2

# 1. cnnmodel
# 2. cnnmodel2
# 3. cnnmodel3
# 4. rnnmodel
# 5. cnnrnnmodel

class cnnmodel(object):
    def __init__(self, config):
        self.TITLE_LEN = config.title_len 
        self.EM_DIM = config.em_dim
        self.beta = config.beta
        self.drop = config.drop
        self.num_filters = config.num_filters
        self.hidden_dim = config.cnn_hidden_dim
        self.trainable = True if config.embed==1 else False

    def __call__(self, embedding_mat):
        TITLE_LEN = self.TITLE_LEN
        EM_DIM = self.EM_DIM
        beta = self.beta
        drop = self.drop
        num_filters = self.num_filters
        hidden_dim = self.hidden_dim
        n_symbols = len(embedding_mat)
        print('---- Embedding Mat Trainable?', self.trainable, '----')

        model_conv2 = Sequential()
        model_conv3 = Sequential()
        model_conv4 = Sequential()

        model_conv2.add(Conv2D(num_filters, kernel_size=(2, EM_DIM), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        model_conv3.add(Conv2D(num_filters, kernel_size=(3, EM_DIM), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        model_conv4.add(Conv2D(num_filters, kernel_size=(4, EM_DIM), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))

        model_conv2.add(BatchNormalization())
        model_conv3.add(BatchNormalization())
        model_conv4.add(BatchNormalization())

        model_conv2.add(MaxPool2D(pool_size=(TITLE_LEN - 1, 1), \
                strides=(1,1), padding='valid'))
        model_conv3.add(MaxPool2D(pool_size=(TITLE_LEN - 2, 1), \
                strides=(1,1), padding='valid'))
        model_conv4.add(MaxPool2D(pool_size=(TITLE_LEN - 3, 1), \
                strides=(1,1), padding='valid'))

        input_a = Input(shape=(TITLE_LEN, ))
        embed = Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
                weights = [embedding_mat], input_length = TITLE_LEN, \
                trainable=self.trainable)(input_a)
        input_reshape = Reshape((TITLE_LEN, EM_DIM, 1))(embed)

        out_1 = model_conv2(input_reshape)
        out_2 = model_conv3(input_reshape)
        out_3 = model_conv4(input_reshape)

        concatenated_tensor = Concatenate(axis=3)([out_1, out_2, out_3])
        flatten = Flatten()(concatenated_tensor)

        dropout = Dropout(drop)(flatten)
        fc = Dense(hidden_dim, kernel_regularizer=L2(beta))(dropout)
        bn = BatchNormalization()(fc)
        flatten = LeakyReLU(0.3)(bn)

        output = Dense(1, activation='sigmoid', kernel_regularizer=L2(beta))(flatten)

        return Model(inputs = input_a, outputs = output)

class cnnmodel2(object):
    def __init__(self, config):
        self.TITLE_LEN = config.title_len 
        self.EM_DIM = config.em_dim
        self.beta = config.beta
        self.drop = config.drop
        self.num_filters = config.num_filters
        self.num_filters2 = config.num_filters2
        self.hidden_dim = config.cnn2_hidden_dim

    def __call__(self):
        TITLE_LEN = self.TITLE_LEN
        EM_DIM = self.EM_DIM
        beta = self.beta
        drop = self.drop
        num_filters = self.num_filters
        num_filters2 = self.num_filters2
        hidden_dim = self.hidden_dim

        model_conv2 = Sequential()
        model_conv3 = Sequential()
        model_conv4 = Sequential()

        # input size: (bs, 65, 100, 1)
        # size: (bs, 64, 1, num_filters)
        model_conv2.add(Conv2D(num_filters, kernel_size=(2, EM_DIM), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 63, 1, num_filters)
        model_conv3.add(Conv2D(num_filters, kernel_size=(3, EM_DIM), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 62, 1, num_filters)
        model_conv4.add(Conv2D(num_filters, kernel_size=(4, EM_DIM), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))

        model_conv2.add(BatchNormalization())
        model_conv3.add(BatchNormalization())
        model_conv4.add(BatchNormalization())
        model_conv2.add(LeakyReLU(0.3))
        model_conv3.add(LeakyReLU(0.3))
        model_conv4.add(LeakyReLU(0.3))

        # size: (bs, 61, 1, num_filters2)
        model_conv2.add(Conv2D(num_filters2, kernel_size=(4, 1), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 61, 1, num_filters2)
        model_conv3.add(Conv2D(num_filters2, kernel_size=(3, 1), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 61, 1, num_filters2)
        model_conv4.add(Conv2D(num_filters2, kernel_size=(2, 1), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))

        model_conv2.add(BatchNormalization())
        model_conv3.add(BatchNormalization())
        model_conv4.add(BatchNormalization())

        model_conv2.add(MaxPool2D(pool_size=(TITLE_LEN - 4, 1), \
                strides=(1,1), padding='valid'))
        model_conv3.add(MaxPool2D(pool_size=(TITLE_LEN - 4, 1), \
                strides=(1,1), padding='valid'))
        model_conv4.add(MaxPool2D(pool_size=(TITLE_LEN - 4, 1), \
                strides=(1,1), padding='valid'))

        input_a = Input(shape=(TITLE_LEN, EM_DIM))
        input_reshape = Reshape((TITLE_LEN, EM_DIM, 1))(input_a)

        out_1 = model_conv2(input_reshape)
        out_2 = model_conv3(input_reshape)
        out_3 = model_conv4(input_reshape)

        concatenated_tensor = Concatenate(axis=3)([out_1, out_2, out_3])
        flatten = Flatten()(concatenated_tensor)

        dropout = Dropout(drop)(flatten)
        fc = Dense(hidden_dim, kernel_regularizer=L2(beta))(dropout)
        bn = BatchNormalization()(fc)
        flatten = LeakyReLU(0.3)(bn)

        output = Dense(1, kernel_regularizer=L2(beta))(flatten)

        return Model(inputs = input_a, outputs = output)

class cnnmodel3(object):
    def __init__(self, config):
        self.TITLE_LEN = config.title_len 
        self.EM_DIM = config.em_dim
        self.beta = config.beta
        self.drop = config.drop

    def __call__(self):
        TITLE_LEN = self.TITLE_LEN
        EM_DIM = self.EM_DIM
        beta = self.beta
        drop = self.drop
        num_filters = 64
        num_filters2 = 128
        num_filters3 = 256
        num_filters4 = 256
        hidden_dim = 128

        model_conv2 = Sequential()
        model_conv3 = Sequential()
        model_conv4 = Sequential()

        # input size: (bs, 65, 100, 1)
        # size: (bs, 64, 1, num_filters)
        model_conv2.add(Conv2D(num_filters, kernel_size=(2, EM_DIM), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 63, 1, num_filters)
        model_conv3.add(Conv2D(num_filters, kernel_size=(3, EM_DIM), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 62, 1, num_filters)
        model_conv4.add(Conv2D(num_filters, kernel_size=(4, EM_DIM), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))

        model_conv2.add(BatchNormalization())
        model_conv3.add(BatchNormalization())
        model_conv4.add(BatchNormalization())
        model_conv2.add(LeakyReLU(0.3))
        model_conv3.add(LeakyReLU(0.3))
        model_conv4.add(LeakyReLU(0.3))

        # size: (bs, 61, 1, num_filters2)
        model_conv2.add(Conv2D(num_filters2, kernel_size=(4, 1), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 61, 1, num_filters2)
        model_conv3.add(Conv2D(num_filters2, kernel_size=(3, 1), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 61, 1, num_filters2)
        model_conv4.add(Conv2D(num_filters2, kernel_size=(2, 1), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))

        model_conv2.add(BatchNormalization())
        model_conv3.add(BatchNormalization())
        model_conv4.add(BatchNormalization())
        model_conv2.add(LeakyReLU(0.3))
        model_conv3.add(LeakyReLU(0.3))
        model_conv4.add(LeakyReLU(0.3))

        # size: (bs, 31, 1, num_filters3)
        model_conv2.add(Conv2D(num_filters3, kernel_size=(3, 1), strides=(2,1), \
                padding='same', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 31, 1, num_filters3)
        model_conv3.add(Conv2D(num_filters3, kernel_size=(3, 1), strides=(2,1), \
                padding='same', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 31, 1, num_filters3)
        model_conv4.add(Conv2D(num_filters3, kernel_size=(3, 1), strides=(2,1), \
                padding='same', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))

        model_conv2.add(BatchNormalization())
        model_conv3.add(BatchNormalization())
        model_conv4.add(BatchNormalization())
        model_conv2.add(LeakyReLU(0.3))
        model_conv3.add(LeakyReLU(0.3))
        model_conv4.add(LeakyReLU(0.3))

        # size: (bs, 15, 1, num_filters4)
        model_conv2.add(Conv2D(num_filters4, kernel_size=(3, 1), strides=(2,1), \
                padding='same', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 15, 1, num_filters4)
        model_conv3.add(Conv2D(num_filters4, kernel_size=(3, 1), strides=(2,1), \
                padding='same', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 15, 1, num_filters4)
        model_conv4.add(Conv2D(num_filters4, kernel_size=(3, 1), strides=(2,1), \
                padding='same', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))

        model_conv2.add(BatchNormalization())
        model_conv3.add(BatchNormalization())
        model_conv4.add(BatchNormalization())
        model_conv2.add(LeakyReLU(0.3))
        model_conv3.add(LeakyReLU(0.3))
        model_conv4.add(LeakyReLU(0.3))

        model_conv2.add(MaxPool2D(pool_size=(15, 1), \
                strides=(1,1), padding='valid'))
        model_conv3.add(MaxPool2D(pool_size=(15, 1), \
                strides=(1,1), padding='valid'))
        model_conv4.add(MaxPool2D(pool_size=(15, 1), \
                strides=(1,1), padding='valid'))

        input_a = Input(shape=(TITLE_LEN, EM_DIM))
        input_reshape = Reshape((TITLE_LEN, EM_DIM, 1))(input_a)

        out_1 = model_conv2(input_reshape)
        out_2 = model_conv3(input_reshape)
        out_3 = model_conv4(input_reshape)

        concatenated_tensor = Concatenate(axis=3)([out_1, out_2, out_3])
        flatten = Flatten()(concatenated_tensor)

        fc = Dense(hidden_dim, kernel_regularizer=L2(beta))(flatten)
        bn = BatchNormalization()(fc)
        flatten = LeakyReLU(0.3)(bn)

        output = Dense(1, kernel_regularizer=L2(beta))(flatten)

        return Model(inputs = input_a, outputs = output)

class rnnmodel(object):
    def __init__(self, config):
        self.TITLE_LEN = config.title_len 
        self.EM_DIM = config.em_dim
        self.beta = config.beta
        self.drop = config.drop
        self.num_filters = config.num_filters
        self.num_filters2 = config.num_filters2
        self.hidden_dim1 = config.rnn_hidden_dim
        self.hidden_dim2 = config.rnn_hidden_dim2
        self.trainable = True if config.embed==1 else False

    def __call__(self, embedding_mat):
        TITLE_LEN = self.TITLE_LEN
        EM_DIM = self.EM_DIM
        beta = self.beta
        drop = self.drop
        num_filters = self.num_filters
        num_filters2 = self.num_filters2
        hidden_dim1 = self.hidden_dim1
        hidden_dim2 = self.hidden_dim2
        n_symbols = len(embedding_mat)
        print('---- Embedding Mat Trainable?', self.trainable, '----')

        # input size: (bs, 65, 100, 1)
        # size: (bs, hidden_dim1, 1)
        model = Sequential([
            Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
                weights = [embedding_mat], input_length = TITLE_LEN, \
                trainable=self.trainable),
            LSTM(hidden_dim1, return_sequences=False, kernel_regularizer=L2(beta)),
            BatchNormalization(),
            LeakyReLU(0.3),
            Dense(hidden_dim2, kernel_regularizer=L2(beta)),
            BatchNormalization(),
            LeakyReLU(0.3),
            Dense(1, activation='sigmoid', kernel_regularizer=L2(beta)),
            ])

        input_a = Input(shape=(TITLE_LEN, ))

        output = model(input_a)

        return Model(inputs = input_a, outputs = output)

class cnnrnnmodel(object):
    def __init__(self, config):
        self.TITLE_LEN = config.title_len 
        self.EM_DIM = config.em_dim
        self.beta = config.beta
        self.drop = config.drop
        self.num_filters = config.num_filters
        self.num_filters2 = config.num_filters2
        self.hidden_dim1 = config.cnnrnn_hidden_dim
        self.hidden_dim2 = config.cnnrnn_hidden_dim2
        self.trainable = True if config.embed==1 else False

    def __call__(self, embedding_mat):
        TITLE_LEN = self.TITLE_LEN
        EM_DIM = self.EM_DIM
        beta = self.beta
        drop = self.drop
        num_filters = self.num_filters
        num_filters2 = self.num_filters2
        hidden_dim1 = self.hidden_dim1
        hidden_dim2 = self.hidden_dim2
        n_symbols = len(embedding_mat)
        print('---- Embedding Mat Trainable?', self.trainable, '----')

        # input size: (bs, 65, 100, 1)
        # size: (bs, 65, hidden_dim1, 1)
        '''
        model = Sequential([
            LSTM(hidden_dim1, return_sequences=True, kernel_regularizer=L2(beta)),
            BatchNormalization(),
            LeakyReLU(0.3),
            Reshape((TITLE_LEN, hidden_dim1, 1)),
            ])
        '''
        model = Sequential()
        model.add(Embedding(output_dim = EM_DIM, input_dim = n_symbols, \
                weights = [embedding_mat], input_length = TITLE_LEN, \
                trainable=self.trainable))
        model.add(LSTM(hidden_dim1, return_sequences=True, kernel_regularizer=L2(beta)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.3))
        model.add(Reshape((TITLE_LEN, hidden_dim1, 1)))

        model_conv2 = Sequential()
        model_conv3 = Sequential()
        model_conv4 = Sequential()

        # size: (bs, 64, 1, num_filters)
        model_conv2.add(Conv2D(num_filters, kernel_size=(2, hidden_dim1), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 63, 1, num_filters)
        model_conv3.add(Conv2D(num_filters, kernel_size=(3, hidden_dim1), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 62, 1, num_filters)
        model_conv4.add(Conv2D(num_filters, kernel_size=(4, hidden_dim1), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))

        model_conv2.add(BatchNormalization())
        model_conv3.add(BatchNormalization())
        model_conv4.add(BatchNormalization())
        model_conv2.add(LeakyReLU(0.3))
        model_conv3.add(LeakyReLU(0.3))
        model_conv4.add(LeakyReLU(0.3))

        # size: (bs, 61, 1, num_filters2)
        model_conv2.add(Conv2D(num_filters2, kernel_size=(4, 1), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 61, 1, num_filters2)
        model_conv3.add(Conv2D(num_filters2, kernel_size=(3, 1), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))
        # size: (bs, 61, 1, num_filters2)
        model_conv4.add(Conv2D(num_filters2, kernel_size=(2, 1), \
                padding='valid', kernel_initializer='normal',\
                kernel_regularizer=L2(beta)))

        model_conv2.add(BatchNormalization())
        model_conv3.add(BatchNormalization())
        model_conv4.add(BatchNormalization())

        model_conv2.add(MaxPool2D(pool_size=(TITLE_LEN - 4, 1), \
                strides=(1,1), padding='valid'))
        model_conv3.add(MaxPool2D(pool_size=(TITLE_LEN - 4, 1), \
                strides=(1,1), padding='valid'))
        model_conv4.add(MaxPool2D(pool_size=(TITLE_LEN - 4, 1), \
                strides=(1,1), padding='valid'))

        input_a = Input(shape=(TITLE_LEN, ))

        out_1 = model(input_a)
        out_2 = model_conv2(out_1)
        out_3 = model_conv3(out_1)
        out_4 = model_conv4(out_1)

        concatenated_tensor = Concatenate(axis=3)([out_2, out_3, out_4])
        flatten = Flatten()(concatenated_tensor)

        dropout = Dropout(drop)(flatten)
        fc = Dense(hidden_dim2, kernel_regularizer=L2(beta))(dropout)
        bn = BatchNormalization()(fc)
        flatten = LeakyReLU(0.3)(bn)

        output = Dense(1, activation='sigmoid', kernel_regularizer=L2(beta))(flatten)

        return Model(inputs = input_a, outputs = output)
