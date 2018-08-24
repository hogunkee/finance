import pickle
from pymongo import MongoClient
import datetime

STOCK_LIST = ['BAC', 'WMT', 'AAPL', 'AMZN', 'MSFT', 'GOOGL', 'FB', 'TSLA', 'NFLX', 'JNJ']

client = MongoClient('localhost', 27017)
db = client['hackathon']
d = '2018-07-09T:00:00:00Z'
#d = datetime.datetime(2018,7,10)

company_data = [[] for i in range(10)]
for i in range(10):
    company = STOCK_LIST[i].lower()
    #rows = db.news.find({'symbol': company}, {'date': 1, 'title': 1, 'text': 1}).sort('date')
    rows = db.news.find({'symbol': company, 'date': {'$lt':d}}, {'date': 1, 'title': 1, 'text': 1}).sort('date')

    for row in list(rows):
        _data = [row['date'].split('T')[0], row['title'], row['text']]
        #_data = list(row.values())[1:]
        company_data[i].append(_data)

result = []
for cd in company_data:
    date, title, text = zip(*cd)
    result.append([date, title, text])
