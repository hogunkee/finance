import pandas as pd
import os
import datetime
import torch
import torch.utils.data as data
import numpy as np
import re
from konlpy.tag import Twitter, Mecab

from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding



def clean_text(text):
    text = text.lower()
    text = re.sub(re.compile('\(.*?\)'), ' ', text)
    text = re.sub(re.compile('\[.*?\]'), ' ', text)
    text = re.sub('[\{\}\[\]\/;:|\)*~`^\-_+<>@\#$\\\=\(\'\"▷■―♡△◇◆㎓㎒,“”·‘’…ㆍ.]', ' ', text)
    '''
    text = re.sub('[↑]', ' ↑', text)
    text = re.sub('[株]', ' 株', text)
    text = re.sub('[弗]', ' 弗', text)
    text = re.sub('[弗]', ' 弗', text)
    '''
    text = " ".join(text.split())
    return text


def morphs(text, max_words, method):
    if method == "basic":
        text = text.split()
    elif method == "twitter":
        twitter = Twitter()
        text = twitter.morphs(text)
    elif method == "mecab":
        mecab = Mecab()
        text = mecab.morphs(text)

    if len(text) > max_words:
        text = text[:max_words]

    text = " ".join(text)

    return text


def morphs_article(input_path, output_path, max_words, method):
    with open(input_path, 'r') as file:
        article_lines = file.readlines()

    # clean articles
    for i, article in enumerate(article_lines):
        article_lines[i] = clean_text(article)

    # tokenize text and limit num words
    for i, article in enumerate(article_lines):
        article_lines[i] = (morphs(article, max_words, method))

    new_lines = "\n".join(article_lines)

    with open(output_path, 'w') as file:
        file.write(new_lines)

    return article_lines


def tokenize_article(input_path, output_path, tokenizer, max_words):
    with open(input_path, 'r') as file:
        article_lines = file.readlines()

    for i, line in enumerate(article_lines):
        article_lines[i] = line.strip()

    lines = tokenizer.texts_to_sequences(article_lines)
    lines = pad_sequences(lines, maxlen=max_words, padding='post')
    lines = np.array(lines)

    np.save(output_path, lines)

    return


def tokenizer_all(data_path, max_words=50, data_type='title', method='twitter'):
    folder_list = os.listdir(data_path)

    article_all = []

    for folder in folder_list:
        if data_type == 'title':
            input_path = os.path.join(data_path, folder, "title.txt")
            output_path = os.path.join(data_path, folder, method + "_title.txt")
            article_all.extend(morphs_article(input_path, output_path, max_words, method))

        elif data_type == 'content':
            input_path = os.path.join(data_path, folder, "content.txt")
            output_path = os.path.join(data_path, folder, method + "_content.txt")
            article_all.extend(morphs_article(input_path, output_path, max_words, method))

    t = Tokenizer()
    t.fit_on_texts(article_all)
    vocab_size = len(t.word_index) + 1
    print("vocabulary size: {}".format(vocab_size))

    for folder in folder_list:
        if data_type == 'title':
            input_path = os.path.join(data_path, folder, method + "_title.txt")
            output_path = os.path.join(data_path, folder, method + "_tokenized_title.npy")
            tokenize_article(input_path, output_path, t, max_words)

        elif data_type == 'content':
            input_path = os.path.join(data_path, folder, method + "_content.txt")
            output_path = os.path.join(data_path, folder, method + "_tokenized_content.npy")
            tokenize_article(input_path, output_path, t, max_words)

    return


if __name__ == "__main__":
    data_path = "data"
    tokenizer_all(data_path=data_path)