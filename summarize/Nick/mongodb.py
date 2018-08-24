import pickle
from pymongo import MongoClient
from reduction import Reduction

STOCK_LIST = ['BAC', 'WMT', 'AAPL', 'AMZN', 'MSFT', 'GOOGL', 'FB', 'TSLA', 'NFLX', 'JNJ']

client = MongoClient('localhost', 27017)
db = client['hackathon']
d = '2018-07-09T:00:00:00Z'

reduction = Reduction()

title_list = []
text_list = []
sum_list = []

for i in range(5,10):
    print(STOCK_LIST[i])
    company = STOCK_LIST[i].lower()
    rows = db.news.find({'symbol': company, 'date': {'$gte':d}}).sort([('date',-1)])
    #rows = db.news.find({'symbol': company, 'date': {'$lt':d}}, {'date': 1, 'title': 1, 'text': 1}).sort('date')

    count = 0
    for row in list(rows):
        count += 1
        if count%20==0:
            print(count)
        title, text, summ = row['title'], row['text'], row['summarize']
        title_list.append(title)
        text_list.append(text)
        sum_list.append(summ)
        #sum_list.append(reduction.reduce(text, 5))

'''
for i in range(len(text_list)):
    db.news.update({'title':title_list[i], 'text':text_list[i]}, \
            {'$set': {'summarize':sum_list[i]}})
'''
