import pickle

def load(name):
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)
