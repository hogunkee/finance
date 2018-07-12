import numpy as np
import pickle

def make_embedding(embedding_dim):
    with open('obj/word2index.pkl', 'rb') as f:
        word2index = pickle.load(f)

    ## new embedding
    n = len(word2index)
    embedding_mat = np.random.normal(0, 0.1, [n, embedding_dim])

    with open('obj/embedding_mat.pkl', 'wb') as f:
        pickle.dump(embedding_mat, f, pickle.HIGHEST_PROTOCOL)

    return
