# XAI 모델 비교 분석을 통한 게임 리뷰 감성 분석 중요 키워드 도출

## 2022년 데이터사이언스학과 졸업논문
- 스팀 게임 리뷰를 감성 분석 한 후 XAI로 비교
- XAI별 상위 키워드 도출
- 키워드 별 가중치 비교, 차이점 비교 분석
- 키워드와 동시에 출현하는 단어 네트워크 분석으로 소비자 니즈 분석


### 프레임워크
![image](https://user-images.githubusercontent.com/70933580/174726081-d1f25395-d8b0-40e2-9edf-b9313774f7f4.png)


### 데이터 수집
- API 크롤링
```python
import requests
import pandas as pd
import os

path_dir = ' ' #불러올 파일들이 있는 위치
file_list = os.listdir(path_dir)

def get_reviews(appid, params={'json':1}): #각 게임별 id 필요
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
    data = pd.read_excel(a)[:20]
    columns=data.columns.values[0]
    number = data[columns].str.split('/').str[4] #id만 따기
    name = data[columns].str.split('/').str[5]
    for i in range(0,len(number)):
        id_ = number[i]   
        response2 = get_n_reviews(str(id_), 5500) #5500은 수집하고자하는 리뷰 갯수 리뷰갯수 조정가능
        df = pd.DataFrame(response2)       
response2 = get_n_reviews(str(1174180), 100000)
df = pd.DataFrame(response2)
df.to_csv("df.csv") #id별 게임 이름으로 
```
###전처리 
