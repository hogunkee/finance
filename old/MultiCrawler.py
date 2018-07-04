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

URL1 = 'https://finance.naver.com/news/news_search.nhn?rcdate=&q=%BB%EF%BC%BA%C0%FC%C0%DA'\
            + '&x=0&y=0&sm=all.basic&pd=1&stDateStart='
URL2 = '&stDateEnd='
BASE_URL = "http://finance.naver.com"


# 기사 검색 페이지에서 기사 제목에 링크된 기사 본문 주소 받아오기
def get_link_from_news_title(page_num, URL, output_dir):
    v = Value('i', 0)
    lock = Lock()

    output_path = ''
    if not os.path.exists(output_dir):
        os.system('mkdir ' + output_dir)

    req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    source_code_from_URL = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')

    num_news = int(soup.find_all('p', 'resultCount')[0].find_all(text=True)[3].replace(',',''))
    print('Total news:', num_news)
    page_num = math.ceil(num_news/20)

    procs = []
    for i in range(page_num):
        print('page: ' + str(i))
        URL_with_page_num = URL + '&page=' + str(i+1)
        req = urllib.request.Request(URL_with_page_num, headers={'User-Agent': 'Mozilla/5.0'})
        source_code_from_URL = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')

        titles = [t for t in soup.find_all('dt', 'articleSubject') + \
                soup.find_all('dd', 'articleSubject')]

        num_process = 4
        num_news = len(titles)
        for i in range(num_process):
            p = Process(target = crawl_with_title_list, \
                    args = (titles[i*5 : (i+1)*5], output_dir, v, lock))
            procs.append(p)
            p.start()
    for p in procs:
        p.join()

    print("총 %d개의 기사를 모았습니다" %v.value)


def crawl_with_title_list(title_list, output_dir, val, lock):
    for t in title_list:
        crawl_with_title(t, output_dir, val, lock)

def crawl_with_title(title, output_dir, val, lock):
    with lock:
        val.value += 1

        if val.value%20==0:
            print('num of news: %d' %val.value)

        news_title = title.find_all(text = True)[1]
        news_title.replace('\n', ' ')
        news_title.replace('\t', ' ')
        news_title.replace('…', ' ')
        news_title = clean_text(news_title)

        title_link = title.select('a')
        article_URL = title_link[0]['href']

        global start_time
        time_interval = time.time() - start_time
        print('[%d] %.3f sec to read \'%s\'' %(val.value, time_interval, news_title))

        if (article_URL[:19] != "/news/news_read.nhn"):
            print("url error")
            return

        news_date, news_text = get_text(BASE_URL + article_URL)
        if not os.path.exists(os.path.join(output_dir, news_date)):
            os.system('mkdir ' + os.path.join(output_dir, news_date))
        output_path = os.path.join(output_dir, news_date)

        file_path = os.path.join(output_path, ('[%03d]'%len(os.listdir(output_path)) + news_title))
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
        string_item.replace('\n', ' ')
        string_item.replace('\t', ' ')
        string_item.replace('…', ' ')
        hangul = re.compile('[^ ㄱ-ㅣ가-힣]+');
        result = hangul.sub('', string_item)

    return news_date.split(' ')[0], clean_text(string_item)

# 클리닝 함수
def clean_text (text):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    cleaned_text = hangul.sub('', text)
    while '  ' in cleaned_text:
        cleaned_text = cleaned_text.replace('  ', ' ')
    '''
    cleaned_text = re.sub ('[a-zA-Z]', '', text)
    cleaned_text = re.sub ('[\{\}\[\]\/?.,;:|\)“‘’”*~`!^\-_+<>@\#$%&\\\=\(\'\"▷■―♡△◇◆]',
                           '', cleaned_text)
    '''
    return cleaned_text


# 메인함수
def main():
    global start_time
    start_time = time.time()

    date_range = pandas.date_range(start='2018-01-01', end='2018-07-01')

    page_num = 10
    output_dir = '../sample'
    for dt in date_range:
        date = str(dt).split(' ')[0]
        target_url = URL1 + date + URL2 + date
        get_link_from_news_title(page_num, target_url, output_dir)

if __name__ == '__main__':
    main()
