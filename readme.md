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

### 전처리 
- Tokenizing, Lemmatization, stopwords 
- 명사만 추출 
- stopwords는 word counting 하면서 업데이트

### 단어카테고리 정의(topic modeling)
- coherencs score에 기반한 토픽수 설정

![image](https://user-images.githubusercontent.com/70933580/174726389-fbebb9d4-71a3-4263-be07-8a28771bd4c2.png)

- 토픽 수 5

![image](https://user-images.githubusercontent.com/70933580/174726508-aa38a293-f8e0-4c87-bc90-8c0bf38a2bf0.png)
- Experience: 게임 플레이와 연관된 단어
- Identity: 게임 디자인에 포함되지만 고유 특성으로 분류될 단어
- Play type: 플레이하는 유형에 관한 단어
- Design: 전반적인 게임 디자인에 관한 단어
- Error: 오류에 관한 단어

### 딥러닝을 통한 감성분석
- 등장 빈도 10회 미만 단어 제거 (7%)
- 길이 8 이하 샘플 제거 (20%)
- 전체 리뷰 길이 분포

![image](https://user-images.githubusercontent.com/70933580/174727000-92674512-da40-412b-b823-baca85972828.png)

- 최대 길이 240
- train: test = 7:3
- 모델 비교

![image](https://user-images.githubusercontent.com/70933580/174727109-7fc9be63-d44c-45cb-a4ec-170c153d69d7.png)

- 명사만 사용한 세 모델의 성능이 비슷
- 세 모델의 가장 기본 모델인 LSTM을 비교 분석에 사용

### XAI 모델 별 해석 및 키워드 비교
- 데이터 필터링 설정
- 모델 별 점수 평균 계산
- 평균 값 IQR 범위 내, 등장빈도 평균 이상 단어 선별
- 필터링 기준

![image](https://user-images.githubusercontent.com/70933580/174727310-ec0e3ed1-68b9-458a-a098-b7323b75e68c.png)

#### LRP 
- score

![image](https://user-images.githubusercontent.com/70933580/174727406-02d53d8e-4533-4336-8dd9-5af18253f1cc.png)

- 등장빈도

![image](https://user-images.githubusercontent.com/70933580/174727428-164448fd-a943-4cd8-9d43-2b30011990b9.png)

#### LIME
- score

![image](https://user-images.githubusercontent.com/70933580/174727493-64453832-9f0a-4064-9afb-b4afa988a311.png)

- 등장빈도

![image](https://user-images.githubusercontent.com/70933580/174727528-df6f40f4-460d-44c6-8679-381f31fa8f47.png)

#### Anchor
- score

![image](https://user-images.githubusercontent.com/70933580/174727618-f80b8dcd-e489-43fd-b97f-2c6b1c9c6bc2.png)

- 등장빈도

![image](https://user-images.githubusercontent.com/70933580/174727652-280fcf02-cb7f-4a35-9cff-839f1547c60e.png)

#### 모델 간 비교
- score

![image](https://user-images.githubusercontent.com/70933580/174727728-d44be3c5-7361-4a8d-b99d-760531e2a5ea.png)

- 등장빈도

![image](https://user-images.githubusercontent.com/70933580/174727754-d65a7ae1-9774-42ee-b693-4a64177a7ee6.png)

### 소비자 의견 클러스터
- 키워드와 매친되는 리뷰 선별
- 선별 리뷰에서 단어 동시출현 네트워크 형성
- 소비자 의견 파악

- 긍정

![image](https://user-images.githubusercontent.com/70933580/174727896-30c63578-0761-4a61-b2b0-5043288dd3ea.png)

- 부정

![image](https://user-images.githubusercontent.com/70933580/174727943-06c466cd-c8da-4c37-8a29-cbb6db68a7da.png)

### 연구 요약 및 의의
- 게임 리뷰 데이터를 딥러닝 모델로 분류
- 분류 결과를 여러 XAI 모델로 해석, 상호 보완적 소비자 의견 도출
- 긍정, 부정에 영향을 미치는 요인들을 정량적으로 도출하기 위한 연구를 진행

#### 기존 연구문제점
- 머신러닝 모델: 딥러닝에 비해 정확도가 떨어짐
- 딥러닝 모델: 설명가능성 부족, 성능향상에만 치중
- XAI 모델 사용: 단일 XAI 모델 사용, 설명 한계

#### 본 연구의 해결 방안
- 딥러닝 모델과 XAI 모델을 결합하여 정확도가 높으면서도 해석력을 높이는 접근 방식 채택
- 다수의 XAI 모델들 간 비교 분석을 통해 해석 결과를 상호보완하고 신뢰도 확보
- 세 모델을 결합하였을 때 어떤 방식을 활용하면 더 좋은 결과를 도찰할 수 있을지에 대한 고찰 수행
- 게임 산업에서의 XAI 적용을 위한 가이드라인 제안

#### 한계점
- 명사 데이터만 사용하여 문맥 고려 되지 않음
- 전체 토큰을 사용한 것에 비해 딥러닝 결과가 상대적으로 좋지않음
- 텍스트 분석 XAI 모델의 한계로 모델 설정에 제한
- 방법론 일반화를 위해 표, 이미지 데이터에도 적용 가능한지 실험
- 게임산업외 다른산업에도 적용가능한지 실험


