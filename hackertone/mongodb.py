from pymongo import MongoClient

STOCK_LIST = ['BAC', 'WMT', 'AAPL', 'AMZN', 'MSFT', 'GOOGL', 'FB', 'TSLA', 'NFLX', 'JNJ']

client = MongoClient('localhost', 27017)
db = client['hackathon']

data = []
count = []
for company in STOCK_LIST:
    company = company.lower()
    rows = db.news.find({'symbol': company}, {'date': 1, 'title': 1})
    data.append(rows)
    count.append(rows.count())
