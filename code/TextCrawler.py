import os
import sys
import time
from bs4 import BeautifulSoup
import urllib.request

import re
from multiprocessing import Pool, Lock, Value, Process
import math
import pandas

sys.setrecursionlimit(100000)

START_DATE = '2015-01-01'
END_DATE = '2018-07-01'

NUM_PROCESS = 32
OUTPUT = '../sample'
URL1 = 'https://finance.naver.com/news/news_search.nhn?rcdate=&q=%BB%EF%BC%BA%C0%FC%C0%DA'\
            + '&x=0&y=0&sm=all.basic&pd=1&stDateStart='
URL2 = '&stDateEnd='
BASE_URL = "http://finance.naver.com"


# 기사 검색 페이지에서 기사 제목에 링크된 기사 본문 주소 받아오기
def pass_url_list(url_list):
    for u in url_list:
        get_link_from_news_title(u)
		
def get_link_from_news_title(URL):
    v = Value('i', 0)
    lock = Lock()

    output_path = ''
    if not os.path.exists(OUTPUT):
        os.system('mkdir ' + OUTPUT)

    req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    source_code_from_URL = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')

    num_news = int(soup.find_all('p', 'resultCount')[0].find_all(text=True)[3].replace(',',''))
    print('Total news:', num_news)
    page_num = math.ceil(num_news/20)

    for i in range(page_num):
        URL_with_page_num = URL + '&page=' + str(i+1)
        req = urllib.request.Request(URL_with_page_num, headers={'User-Agent': 'Mozilla/5.0'})
        source_code_from_URL = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')

        titles = [t for t in soup.find_all('dt', 'articleSubject') + \
                soup.find_all('dd', 'articleSubject')]

        num_process = 4
        num_news = len(titles)
		
        for title in titles:
            crawl_with_title(title, v, lock)

    print("총 %d개의 기사를 모았습니다" %v.value)


'''
def crawl_with_title_list(title_list, val, lock):
    for t in title_list:
        crawl_with_title(t, val, lock)
'''

def crawl_with_title(title, val, lock):
    with lock:
        val.value += 1

        news_title = title.find_all(text = True)[1]
        news_title.replace('\n', ' ')
        news_title.replace('\t', ' ')
        news_title.replace('…', ' ')
        news_title = clean_text(news_title)

        title_link = title.select('a')
        article_URL = title_link[0]['href']

        if (article_URL[:19] != "/news/news_read.nhn"):
            print("url error")
            return

        news_date, news_text = get_text(BASE_URL + article_URL)
        if not os.path.exists(os.path.join(OUTPUT, news_date)):
            os.system('mkdir ' + os.path.join(OUTPUT, news_date))
        output_path = os.path.join(OUTPUT, news_date)

        global start_time
        time_interval = time.time() - start_time
        print('<%s> [%d] %.3f sec. now reading \'%s\'' \
                %(news_date, val.value, time_interval, news_title))
        #print('[%d] %.3f sec to read \'%s\'' %(val.value, time_interval, news_title))

        file_path = os.path.join(output_path, news_title)
        #file_path = os.path.join(output_path, ('%03d-'%len(os.listdir(output_path)) + news_title))
        if not os.path.exists(file_path):
            output_file = open(file_path, 'w', encoding='utf-8')
            output_file.write(news_text)
            output_file.close() 

# 기사 본문 내용 긁어오기 (위 함수 내부에서 기사 본문 주소 받아 사용되는 함수)
def get_text(URL):
    news_date = ''
    URL = URL[:98]+"%BB%EF%BC%BA%C0%FC%C0%DA"+URL[102:]
    source_code_from_URL = urllib.request.urlopen(URL).read()
    soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
    for date in soup.find_all('span', 'article_date'):
        news_date=date.find_all(text = True)[0]
    if news_date=='':
        news_date = '2018-07-02 00:00:00'
    for item in soup.find_all('div', id = 'content'):
        string_item = (item.find_all(text=True))
        string_item = ' '.join(string_item).strip()
        string_item = string_item[:string_item.rfind('@')]
        if string_item.find('▶')>20:
            string_item = string_item[:string_item.find('▶')]

    return news_date.split(' ')[0], clean_text(string_item)

# 클리닝 함수
def clean_text (text):
    hangul = re.compile('[^ a-zA-Z0-9ㄱ-ㅣ가-힣]+')
    text = hangul.sub(' ', text)
    while '  ' in text:
        text = text.replace('  ', ' ')
    while '[' in text and ']' in text and text.count('[')==text.count(']'):
        text = text.split('[')[0] + text.split(']')[1]
    while '(' in text and ')' in text and text.count('(')==text.count(')'):
        text = text.split('(')[0] + text.split(')')[1]
    '''
    cleaned_text = re.sub ('[a-zA-Z]', '', text)
    cleaned_text = re.sub ('[\{\}\[\]\/?.,;:|\)“‘’”*~`!^\-_+<>@\#$%&\\\=\(\'\"▷■―♡△◇◆]',
                           '', cleaned_text)
    '''
    return text.strip()


# 메인함수
def main():
    global start_time
    start_time = time.time()

    start_date = '2018-01-07'
    end_date = '2018-07-01'
    date_range = pandas.date_range(start=START_DATE, end=END_DATE)
    url_list = list(map(lambda d: URL1+str(d).split(' ')[0]+URL2+str(d).split(' ')[0],\
        date_range))

    proc_url_list = []
    for i in range(NUM_PROCESS):
        proc_url_list.append(url_list[i::NUM_PROCESS])

    procs = []
    for i in range(NUM_PROCESS):
        p = Process(target = pass_url_list, args = (proc_url_list[i],))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

if __name__ == '__main__':
    main()
