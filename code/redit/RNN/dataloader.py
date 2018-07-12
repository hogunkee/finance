import os
import numpy as np
import pandas as pd

GLOVE_DIR = '../data/glove.6B/'
CSV_DIR = '../data/Full_Data.csv'
EM_DIM = 100

def load_data(csv_dir):
    data = pd.read_csv(csv_dir, encoding='ISO-8859-1')

    train = data[data['Date'] < '20150101']
    test = data[data['Date'] > '20141231']

    # Removing punctuations
    traindata= train.iloc[:,2:27]
    testdata= test.iloc[:,2:27] 
    traindata.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)
    testdata.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)

    # Renaming column names for ease of access
    list1= [i for i in range(25)]
    new_Index=[str(i) for i in list1]
    traindata.columns= new_Index
    testdata.columns= new_Index

    # Convertng headlines to lower case
    for index in new_Index:
        traindata[index]=traindata[index].str.lower()
        testdata[index]=testdata[index].str.lower()

    train_concat = []
    test_concat = []
    for row in range(0,len(traindata.index)):
        train_concat.append(' '.join(str(x) for x in traindata.iloc[row,0:25]))
    for row in range(0,len(testdata.index)):
        test_concat.append(' '.join(str(x) for x in testdata.iloc[row,0:25]))

    print(type(train['Label'].values))
    return (train_concat, train['Label'].values.tolist()), \
            (test_concat, test['Label'].values.tolist())

def make_embedding(tokenizer, voca_size):
    pre_embedding = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.'+str(EM_DIM)+'d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        pre_embedding[word] = coefs
    f.close()

    word2index = tokenizer.word_index
    n_symbols = min(voca_size, len(word2index))

    ## new embedding
    x,y=0,0
    embedding_mat = np.zeros((n_symbols, EM_DIM))
    for _word, _idx in word2index.items():
        if _idx < n_symbols:
            if _word in pre_embedding.keys():
                embedding_mat[_idx] = pre_embedding[_word]
                x+=1
            else:
                embedding_mat[_idx] = np.random.normal(0, 0.1, EM_DIM)
                y+=1
    print('embedding:', x)
    print('no matching:', y)

    return embedding_mat


