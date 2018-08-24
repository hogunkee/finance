import os
import pandas

START_DATE = '2015-01-01'
END_DATE = '2018-07-01'
OUTPUT = '../sample'
date_range = pandas.date_range(start=START_DATE, end=END_DATE)
date_list = list(map(lambda d: str(d).split(' ')[0], date_range))

exist_date = os.listdir(OUTPUT)
target_list = list(set(date_list) - set(exist_date))
print(sorted(target_list))
print(len(target_list))
