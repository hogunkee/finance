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


def date_generator(start_date, days, reverse=True):
    date_list = []
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    if reverse:
        for i in range(days):
            day = start_date - datetime.timedelta(i + 1)
            day = day.strftime("%Y-%m-%d")
            date_list.append(day)
    else:
        for i in range(days):
            day = start_date + datetime.timedelta(i + 1)
            day = day.strftime("%Y-%m-%d")
            date_list.append(day)

    return date_list


class Dataset(data.Dataset):
    def __init__(self, data_path, stock, num_past_days, max_articles, max_words, \
                 token_method, use_title=True):
        self.data_path = data_path
        self.stock = stock
        self.num_past_days = num_past_days
        self.use_title = use_title
        self.max_articles = max_articles
        self.max_words = max_words
        self.token_method = token_method

        self.date_list = stock["date"][:-1]
        self.labels = np.float32(np.diff(stock["value"]) / stock["value"][:-1])
        self.labels = np.float32(self.labels > 0)
        self.date_labels = pd.DataFrame(self.labels,
                                        index=self.date_list)
        self.build_dataset()

    def build_dataset(self):

        first = 0
        for date in self.date_list:
            prev_date = datetime.datetime.strptime(date, "%Y-%m-%d") - datetime.timedelta(days=1)
            prev_date = prev_date.strftime("%Y-%m-%d")
            file_path = os.path.join(self.data_path, prev_date, self.token_method
                                     + "_tokenized_title.npy")
            file = np.load(file_path)
            file = file[:, :self.max_words]
            label = self.date_labels.loc[date,0]
            label = label * np.ones(len(file))


            if first == 0 :
                self.dataset = file
                self.label_dataset = label
                first += 1
            else:
                self.dataset = np.vstack((self.dataset, file))
                self.label_dataset = np.append(self.label_dataset, label)
        self.dataset = torch.LongTensor(self.dataset)
        self.label_dataset = torch.FloatTensor(self.label_dataset)

    def __getitem__(self, index):

        return self.dataset[index], self.label_dataset[index]

    def __len__(self):
        return len(self.label_dataset)


def get_loader(data_path="data", stock, batch_size=10, num_workers=2, num_past_days=3, \
               max_articles=20, max_words=20, token_method='twitter'):
    dataset = Dataset(data_path, stock, num_past_days, max_articles, max_words, token_method)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader

if __name__ == "__main__":
    loader = get_loader("data", "stock.csv")
    for i, data in enumerate(loader):
        print(data[0].shape)
        print(data[1])
        if i == 10:
            break