# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:24:45 2021

@author: young
"""

#패키지 설치
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
        
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from selenium.common.exceptions import ElementNotVisibleException
import re
import requests as res
import time
import random
import openpyxl
import os
import xmltodict 
from urllib.request import urlopen 
import json
import datetime
from datetime import datetime
from selenium.webdriver.common.keys import Keys

#드라이버 설정
driver = webdriver.Chrome('./chromedriver')

#메타크리틱 리뷰 크롤링
user_name=[]
score =[]
date =[]
review=[]

for a in range(0,4):
    url ='https://www.metacritic.com/game/playstation-4/grand-theft-auto-v/user-reviews?sort-by=date&num_items=100&page='+str(a)
    driver.get(url)
    time.sleep(3)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')   

    names = soup.find_all('div', {'class','name'})
    scores = soup.find_all('div', {'class','review_grade'})
    dates = soup.find_all('div', {'class','date'})
    reviews = soup.find_all('div', {'class','review_body'})
    
    expend = driver.find_elements_by_css_selector('span.toggle_expand_collapse.toggle_expand')
    for a in expend:
        a.click()
        
    for i in names:
        user_name.append(i.text.strip())
    for j in scores:
        score.append(j.text.strip())
    for k in dates:
        date.append(k.text)
    for l in reviews:
        review.append(l.text.lstrip())

meta_review = pd.DataFrame([user_name, score, date, review]).T

#2020년 이전 리뷰 제거 
meta = meta_review.iloc[0:355,:]

#파일 저장
meta.to_excel('meta_review.xlsx')


genre = ['action', 'rpg', 'strategy', 'adventure_and_casual', 'simulation', 'sports_and_racing']

link_list=[]
for j in genre:
    for i in range(2):
        url ="https://store.steampowered.com/category/"+j+"/#p="+str(i)+"&tab=TopRated"
        driver.get(url)
        driver.get(url)
        time.sleep(2)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')   
        link_list.append(j)
        for link in soup.find('div', {'class': 'peeking_carousel', 'id': 'TopRatedRows'}).find_all('a'):            
            link_list.append(link.attrs['href'])

link_list_df = pd.DataFrame(link_list)
link_list_df.drop_duplicates(inplace=True)

link_list_df.to_excel('C:/Users/young/OneDrive/바탕 화면/link_list_df.xlsx')

url2 = "https://store.steampowered.com/app/730/CounterStrike_Global_Offensive/?snr=1_241_4_action_tab-TopRated"

driver.get(url2)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')   




# 스크롤 높이 가져옴
path_dir = 'C:/Users/young/OneDrive/바탕 화면/졸업논문/data/새 폴더'
file_list = os.listdir(path_dir)



for a in file_list:    
    data = pd.read_excel("C:/Users/young/OneDrive/바탕 화면/졸업논문/data/새 폴더/"+a)[:20]
    columns = data.columns.values[0]
    number = data[columns].str.split('/').str[4] #id만 따기    
    for k in number:
        url = 'https://steamcommunity.com/app/'+str(k)+'/reviews/?p=1&browsefilter=trendthreemonths' #리뷰 사이트 접속   
        driver.get(url) #드라이브 열기         
        time.sleep(2)
        try:
            driver.find_element_by_xpath('/html/body/div[1]/div[7]/div[9]/div/div[1]/div[1]').click()
            time.sleep(4)
        except Exception:
            pass

        #필요한 만큼 스크롤 후 정지 

        
        last_height = driver.execute_script("return document.body.scrollHeight")
        start = datetime.datetime.now()
        end = start + datetime.timedelta(seconds=3600)    
        while True:
            
            # 끝까지 스크롤 다운
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
            # 2초 대기
            SCROLL_PAUSE_SEC = 3
            time.sleep(SCROLL_PAUSE_SEC)
        
            # 스크롤 다운 후 스크롤 높이 다시 가져옴
            new_height = driver.execute_script("return document.body.scrollHeight")
            time.sleep(SCROLL_PAUSE_SEC)
            
            if new_height == last_height:
                break
            last_height = new_height
            if datetime.datetime.now() > end:
                break
        #필요한 만큼 스크롤한 후 중지    
        time.sleep(3)
        #스팀 리뷰 크롤링
        user_name=[]
        score =[]
        review=[]
        
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')   
        
        title = soup.find('div', {'clsaa', 'apphub_AppName ellipsis'}).text
        names = soup.find_all('div', {'class','apphub_CardContentAuthorName offline ellipsis'})
        scores = soup.find_all('div', {'class','title'})
        reviews = soup.find_all('div', {'class', 'apphub_CardTextContent'})
        

        for i in names:
            user_name.append(i.text.strip())
        for j in scores:
            score.append(j.text.strip())
        for l in reviews:
            review.append(l.text)
            
        steam_review = pd.DataFrame([user_name, score,review]).T
        steam_review.columns=['user_name', 'score', 'review']
        
        #replace 사용 불필요한 단어 제거 
        steam_review['review'] = steam_review['review'].str.replace('Posted', '')

        #정규 표현식 사용 불필요한 단어 제거
        sample = []
        for i in steam_review['review'].values.tolist():
            sample.append(re.sub("([A-Z]+)([':'])([' '])([0-9]{0,2})([' '])([A-Z][a-z]+)", '',i))
        
        sample2 =[]
        for i in sample:
            sample2.append(re.sub("([A-Z][a-z]+[' '][0-9]{0,2})", '', i))        
        
        steam_review['review'] = sample2
        
        #공백 제거
        steam_review['review'] = steam_review['review'].str.strip()  
        
        #파일로 저장
        steam_review.to_excel('C:/Users/young/OneDrive/바탕 화면/졸업논문/data/'+str(columns)+title +'.xlsx')


#API를 써서 

import requests
import pandas as pd
import os


path_dir = 'C:/Users/young/OneDrive/바탕 화면/졸업논문/data/새 폴더'
file_list = os.listdir(path_dir)

def get_reviews(appid, params={'json':1}):
        url = 'https://store.steampowered.com/appreviews/'
        response = requests.get(url=url+appid, params=params)
        return response.json()

def get_n_reviews(appid, n):
    reviews = []
    cursor = '*'
    params = {
            'json' : 1,
            'filter' : 'all',
            'language' : 'english',
            'day_range' : 9223372036854775807,
            'review_type' : 'all',
            'purchase_type' : 'all'
            }

    while n > 0:
        params['cursor'] = cursor.encode()
        params['num_per_page'] = min(100, n)
        n -= 100

        response = get_reviews(appid, params)
        cursor = response['cursor']
        reviews += response['reviews']

        if len(response['reviews']) < 100: break

    return reviews



for a in file_list[2:]:
    data = pd.read_excel("C:/Users/young/OneDrive/바탕 화면/졸업논문/data/새 폴더/"+a)[:20]
    columns=data.columns.values[0]
    number = data[columns].str.split('/').str[4] #id만 따기
    name = data[columns].str.split('/').str[5]
    for i in range(0,len(number)):
        id_ = number[i]
       
        response2 = get_n_reviews(str(435150), 5500)
        df = pd.DataFrame(response2)
        df.to_csv("C:/Users/young/OneDrive/바탕 화면/졸업논문/data/"+"adventure_and_casual"+'@'+'Divinity_Original_Sin_2__Definitive_Edition'+".csv")

response2 = get_n_reviews(str(1174180), 100000)
df = pd.DataFrame(response2)
