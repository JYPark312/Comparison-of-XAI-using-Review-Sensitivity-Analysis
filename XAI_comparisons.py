# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:24:48 2022

@author: young
"""
import pandas as pd

import plotly.express as px
from plotly.offline import plot

### LIME 비교 분석
#긍정
sum_p = pd.read_csv('라임_긍정_가중치.csv', index_col=0)

sum_p = sum_p.reset_index()
sum_p = sum_p.rename(columns={'word': 'count', 'index': 'word'})

sum_p.describe()

sum_p= sum_p[(sum_p['value'] >= 0.016125) & (sum_p['value'] <= 0.081000)] #값 outlier 제거 
sum_p= sum_p[sum_p['count'] >= 3.908068 ] #횟수 평균이상

#부정
sum_n = pd.read_csv('라임_부정_가중치.csv', index_col=0)

sum_n = sum_n.reset_index()
sum_n = sum_n.rename(columns={'word': 'count', 'index': 'word'})

sum_n= sum_n[(sum_n['value'] >= -0.119000) & (sum_n['value'] <= -0.023000)] #값 outlier 제거 
sum_n= sum_n[sum_n['count'] >= 2.699561 ] #횟수 평균이상

# fig = px.scatter(sum_n, x='count', y='value')
# plot(fig, auto_open=True)

### anchor 비교 분석
anchor_df = pd.read_csv('./anchor_result.csv', index_col=0)

anchor_p = anchor_df[anchor_df['mean']>=0.5]
anchor_n = anchor_df[anchor_df['mean'] < 0.5]

anchor_p = pd.concat([anchor_p['mean'].groupby(anchor_p['names']).mean(), anchor_p['names'].value_counts()], axis=1)
anchor_n = pd.concat([anchor_n['mean'].groupby(anchor_n['names']).mean(), anchor_n['names'].value_counts()], axis=1)

anchor_p, anchor_n = anchor_p.reset_index(), anchor_n.reset_index() 
anchor_p, anchor_n = anchor_p.rename(columns={'names': 'count', 'index': 'word'}), anchor_n.rename(columns={'names': 'count', 'index': 'word'})

# 긍정 분석
anchor_p.describe()

anchor_p = anchor_p[(anchor_p['mean'] >= 0.995773) & (anchor_p['mean']<=1.000000)]
anchor_p = anchor_p[(anchor_p['count'] >= 1.835938)]

#부정 분석
anchor_n.describe()

anchor_n = anchor_n[(anchor_n['mean'] >= 0.100971) & (anchor_n['mean']<=0.101658)]
anchor_n = anchor_n[anchor_n['count'] >= 1.710120]


#### LRP 비교 분석
#긍정
lrp_p = pd.read_csv('./positive_lrp.csv', index_col=0)

lrp_p = lrp_p[(lrp_p['lrp_rele'] >0) & (lrp_p['positive_sa'] >0)]
lrp_p = pd.concat([lrp_p['positive_sa'].groupby(lrp_p['words']).mean(), lrp_p['lrp_rele'].groupby(lrp_p['words']).mean(), lrp_p['words'].value_counts()], axis=1)

lrp_p = lrp_p.reset_index()
lrp_p = lrp_p.rename(columns={'words': 'count', 'index': 'word'})

lrp_p.describe()

lrp_p = lrp_p[(lrp_p['lrp_rele'] >= 0.015900) & (lrp_p['lrp_rele']<=0.221000) & (lrp_p['count'] >= 2.081081)]


#부정
lrp_n = pd.read_csv('./negative_lrp.csv', index_col=0)

lrp_n = lrp_n[(lrp_n['lrp_rele'] > 0) & (lrp_n['positive_sa'] > 0)]
lrp_n = pd.concat([lrp_n['positive_sa'].groupby(lrp_n['words']).mean(), lrp_n['lrp_rele'].groupby(lrp_n['words']).mean(), lrp_n['words'].value_counts()], axis=1)

lrp_n.describe()

lrp_n = lrp_n[(lrp_n['lrp_rele'] >= 0.018842) & (lrp_n['lrp_rele']<=0.324100) & (lrp_n['count'] >= 1.708978)]

lrp_p.describe()

sum_n.to_csv('lime_negative.csv')
sum_p.to_csv('lime_positive.csv')
anchor_n.to_csv('anchor_negative.csv')
anchor_p.to_csv('ahcnor_positive.csv')
lrp_n.to_csv('lrp_negative.csv')
lrp_p.to_csv('lrp_positive.csv')