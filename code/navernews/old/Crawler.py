import os
import time
import sys
from bs4 import BeautifulSoup
import urllib.request

import re
from urllib.parse import quote
import pickle


TARGET_URL = 'https://finance.naver.com/news/news_search.nhn?rcdate=&q=%BB%EF%BC%BA%C0%FC%C0%DA\
            &x=0&y=0&sm=all.basic&pd=1&stDateStart=1997-01-01&stDateEnd=2018-07-02&page='
#TARGET_URL = "http://finance.naver.com/news/news_search.nhn?rcdate=&q=%BB%EF%BC%BA%C0%FC%C0%DA\
#&x=0&y=0&sm=all.basic&pd=1&stDateStart=1997-01-01&stDateEnd=2018-06-18&page="
BASE_URL = "http://finance.naver.com"

# 기사 검색 페이지에서 기사 제목에 링크된 기사 본문 주소 받아오기
def get_link_from_news_title(page_num, URL, output_dir):
    output_path = ''
    old_date = ''
    count = 0
    if not os.path.exists(output_dir):
        os.system('mkdir ' + output_dir)

    for i in range(page_num):
        print('page: ' + str(i))
        URL_with_page_num = URL + str(i+1)
        req = urllib.request.Request(URL_with_page_num, headers={'User-Agent': 'Mozilla/5.0'})
        source_code_from_URL = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')

        start_time = time.time()
        real_start_time = start_time
        for title in soup.find_all('dt', 'articleSubject')+soup.find_all('dd', 'articleSubject'):
            news_title = title.find_all(text = True)[1]
            #print(str(count) + ': '+ news_title)
            title_link = title.select('a')
            article_URL = title_link[0]['href']

            end_time = time.time()
            time_interval = end_time - start_time
            cumulative_time = end_time - real_start_time
            start_time = end_time
            print('[%d] %.3f sec (%.3f sec) to read <%s>' %(count, cumulative_time,\
                time_interval, news_title))

            if (article_URL[:19] != "/news/news_read.nhn"):
                print("url error")
                return
            count+=1
            if count%5==0:
                print('num of news: %d' %count)

            news_date, news_text = get_text(BASE_URL + article_URL)
            if old_date != news_date:
                old_date = news_date
                if not os.path.exists(os.path.join(output_dir, news_date)):
                    print('News Date: ' + os.path.join(output_dir, news_date))
                    os.system('mkdir ' + os.path.join(output_dir, news_date))
                output_path = os.path.join(output_dir, news_date)

            file_path = os.path.join(output_path, news_title)
            output_file = open(file_path, 'w', encoding='utf-8')
            output_file.write(news_text)
            output_file.close()

    print("총 %d개의 기사를 모았습니다" %count)


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
        '''
        if string_item.find('▶')>20:
            string_item = string_item[:string_item.find('▶')]
        string_item.replace('\n', ' ')
        string_item.replace('\t', ' ')
        '''
    return news_date.split(' ')[0], string_item

# 클리닝 함수
def clean_text (text):
    cleaned_text = re.sub ('[a-zA-Z]', '', text)
    cleaned_text = re.sub ('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"▷■―♡△◇◆]',
                           '', cleaned_text)
    return cleaned_text


# 메인함수
def main():
    page_num = 10
    output_dir = '../sample'
    get_link_from_news_title(page_num, TARGET_URL, output_dir)

    '''
    INPUT_FILE_NAME = "out.txt"
    OUTPUT_FILE_NAME = "clean_out.txt"
    read_file = open(INPUT_FILE_NAME, 'r', encoding='utf-8')
    write_file = open(OUTPUT_FILE_NAME, 'w', encoding='utf-8')
    text = read_file.read()
    text = clean_text(text)
    write_file.write(text)
    read_file.close()
    write_file.close()
    '''

if __name__ == '__main__':
    main()
