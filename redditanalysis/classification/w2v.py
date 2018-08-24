from data_loader import *
from gensim.test.utils import common_texts
from gensim.models import FastText
from gensim.test.utils import get_tmpfile

def split_title(title):
    return title.split()

def fasttext():
    f_name = 'obj/fasttext.model'
    if os.path.exists(f_name):
        model = FastText.load(f_name)

    else:
        df = load_data()
        sentences = df['title'].apply(split_title).values

        model = FastText(sentences, min_count=1)
        model.save(f_name)

    #embed_mat = model.wv.vectors
    return model
