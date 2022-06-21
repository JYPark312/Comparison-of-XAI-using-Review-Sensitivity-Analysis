# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:43:41 2022

@author: young
"""
import keras 
from numpy import newaxis as na
import pandas as pd
import numpy as np
import re

import os
from tqdm import tqdm
import string
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer

###LRP class setting
class LRP4LSTM(object):
    def __init__(self, model):
        self.model = model
        
        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights() #모델 가중치 호출

        # suppress scientific notation 내 모델 이름으로 수정
        np.set_printoptions(suppress=True)
        for name, weight in zip(names, weights):
            if name == 'lstm/lstm_cell/kernel:0':
                kernel_0 = weight
            if name == 'lstm/lstm_cell/recurrent_kernel:0':
                recurrent_kernel_0 = weight
            if name == 'lstm/lstm_cell/bias:0':
                bias_0 = weight
            elif name == 'dense/kernel:0':
                output = weight


        print("kernel_0", kernel_0.shape)
        print("recurrent_kernel_0", recurrent_kernel_0.shape)
        print("bias_0", bias_0.shape)
        print("output", output.shape)

        # self.Wxh_Left (240, 60)
        # self.Whh_Left (240, 60)
        # self.bxh_Left (240,)
        # self.Why_Left (5, 60)

        self.Wxh = kernel_0.T  # shape 4d*e
        self.Whh = recurrent_kernel_0.T  # shape 4d
        self.bxh = bias_0.T  # shape 4d 
        self.Why = output.T
        
    def lrp_linear(self, hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor=1.0, debug=False):
        """
        LRP for a linear layer with input dim D and output dim M.
        Args:
        - hin:            forward pass input, of shape (D,)
        - w:              connection weights, of shape (D, M)
        - b:              biases, of shape (M,)
        - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
        - Rout:           relevance at layer output, of shape (M,)
        - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution is redistributed for sanity check)
        - eps:            stabilizer (small positive number)
        - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
        Returns:
        - Rin:            relevance at layer input, of shape (D,)
        """
        sign_out = np.where(hout[na,:]>=0, 1., -1.) # shape (1, M)

        numer    = (w * hin[:,na]) + ( bias_factor * (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units ) # shape (D, M)
        # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
        # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
        # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)

        denom    = hout[na,:] + (eps*sign_out*1.)   # shape (1, M)

        message  = (numer/denom) * Rout[na,:]       # shape (D, M)

        Rin      = message.sum(axis=1)              # shape (D,)

        if debug:
            print("local diff: ", Rout.sum() - Rin.sum())
        # Note: 
        # - local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D (i.e. when only one incoming layer)
        # - global network relevance conservation if bias_factor==1.0 and bias_nb_units set accordingly to the total number of lower-layer connections 
        # -> can be used for sanity check

        return Rin
        
    def get_layer_output(self, layer_name, data): #layer의 아웃풋
        # https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
        intermediate_layer_model = keras.Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer_name).output)
        return intermediate_layer_model.predict(data)  
    
    def run(self, target_data, target_class):
        def sigmoid(x): 
            return 1 / (1 + np.exp(-x))
        
        #원본 소스에서 E embedding은 전체에 대한 단어 사전이고, x는 embedding된 인풋이다.  
        
        # x = self.get_layer_output('embedding', target_data).squeeze(axis=1)
        x = self.get_layer_output('embedding', target_data)
        e = x.shape[1]

       ################# forword
        T = target_data.shape[0]
        d = int(512/4)  # hidden units
        C = self.Why.shape[0] # number of classes

        idx    = np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) # indices of gates i,f,o together
        idx_i, idx_f, idx_c, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,g,f,o separately

        # 최종적으로 구하려는 값은 c에 저장될 값과 h으로 지워질 값
        h  = np.zeros((T,d))
        c  = np.zeros((T,d))

        gates_pre = np.zeros((T, 4*d))  # gates pre-activation
        gates     = np.zeros((T, 4*d))  # gates activation

        for t in range(T):

            gates_pre[t]    = np.dot(self.Wxh, x[t]) + np.dot(self.Whh, h[t-1]) + self.bxh

            gates[t,idx]    = sigmoid(gates_pre[t,idx])
            gates[t,idx_c]  = np.tanh(gates_pre[t,idx_c]) 

            c[t]            = gates[t,idx_f]*c[t-1] + gates[t,idx_i]*gates[t,idx_c]
            h[t]            = gates[t,idx_o]*np.tanh(c[t])

        score = np.dot(self.Why, h[t])    

        ################# backwork
        dx     = np.zeros(x.shape)

        dh          = np.zeros((T, d))
        dc          = np.zeros((T, d))
        dgates_pre  = np.zeros((T, 4*d))  # gates pre-activation
        dgates      = np.zeros((T, 4*d))  # gates activation

        ds               = np.zeros((C))
        ds[target_class] = 1.0
        dy               = ds.copy()

        #맨처음을 0으로 시작하지 않게 위한조치
        dh[T-1]     = np.dot(self.Why.T, dy)

        for t in reversed(range(T)):  #T를 뒤집어서 마지막 층부터 처음 층까지 역으로 가면서 계산
            dgates[t,idx_o]    = dh[t] * np.tanh(c[t])  # do[t]
            dc[t]             += dh[t] * gates[t,idx_o] * (1.-(np.tanh(c[t]))**2) # dc[t]
            dgates[t,idx_f]    = dc[t] * c[t-1]         # df[t]
            dc[t-1]            = dc[t] * gates[t,idx_f] # dc[t-1]
            dgates[t,idx_i]    = dc[t] * gates[t,idx_c] # di[t]
            dgates[t,idx_c]    = dc[t] * gates[t,idx_i] # dg[t]
            dgates_pre[t,idx]  = dgates[t,idx] * gates[t,idx] * (1.0 - gates[t,idx]) # d ifo pre[t]
            dgates_pre[t,idx_c]= dgates[t,idx_c] *  (1.-(gates[t,idx_c])**2) # d c pre[t]
            dh[t-1]            = np.dot(self.Whh.T, dgates_pre[t])
            dx[t]              = np.dot(self.Wxh.T, dgates_pre[t])

        ################# LRP
        eps=0.001 
        bias_factor=1.0
        Rx  = np.zeros(x.shape)
        Rh  = np.zeros((T+1, d))
        Rc  = np.zeros((T+1, d))
        Rg  = np.zeros((T,   d)) # gate g only

        Rout_mask            = np.zeros((C))
        Rout_mask[target_class] = 1.0

        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        Rh[T-1]  = self.lrp_linear(h[T-1], self.Why.T, np.zeros((C)), score, score*Rout_mask, d, eps, bias_factor, debug=False)  

        for t in reversed(range(T)): 
            Rc[t]   += Rh[t]
            Rc[t-1]  = self.lrp_linear(gates[t,idx_f]*c[t-1], np.identity(d), np.zeros((d)), c[t], Rc[t], d, eps, bias_factor, debug=False)
            Rg[t]    = self.lrp_linear(gates[t,idx_i]*gates[t,idx_c], np.identity(d), np.zeros((d)), c[t], Rc[t], d, eps, bias_factor, debug=False)
            Rx[t]    = self.lrp_linear(x[t], self.Wxh[idx_c].T, self.bxh[idx_c], gates_pre[t,idx_c], Rg[t], d+e, eps, bias_factor, debug=False)
            Rh[t-1]  = self.lrp_linear(h[t-1], self.Whh[idx_c].T, self.bxh[idx_c], gates_pre[t,idx_c], Rg[t], d+e, eps, bias_factor, debug=False)    

        return score, x, dx, Rx, Rh[-1].sum()
    
### 랜덤으로 샘플 1000개 추출    
import random  

number_list = list(range(0,53101))
number_list = random.sample(number_list, 1000)

np_list = pd.DataFrame()    

for i in number_list:
    np_list2 = pre_data[pre_data.index.values == i]
    np_list = pd.concat([np_list, np_list2])
    
#########################################LRP####################################
lrp = LRP4LSTM(model)

def int_to_str(target_class):
    if target_class == 0 :
        return "부정"
    else :
        return "긍정"

def index_to_word(list):
    _ = []
    for x in list :
        _.append(words_list[x])
    return _

positive_list = np_list[np_list['label'] ==1]
negative_list = np_list[np_list['label'] ==0]

import warnings
warnings.filterwarnings("ignore")

###LRP 긍정 리뷰 분석
po_R_words_SA= pd.DataFrame()
po_R_words= pd.DataFrame()
po_words = pd.DataFrame()

for i in tqdm(range(len(positive_list))):
    try:
        list_tokenized_ex = tokenizer3.texts_to_sequences([positive_list['review_text'].iloc[i]])
        target_full_data = pad_sequences(list_tokenized_ex, maxlen=max_review_length)
        
        target_data = target_full_data[target_full_data != 0]
        target_class = np.argmax(np_list['label'].iloc[i])
        
        score, x, Gx, Rx, R_rest = lrp.run(target_data, target_class)    
        
        po_R_words_SA=pd.concat([po_R_words_SA, pd.DataFrame(((np.linalg.norm(Gx, ord=2, axis=1))**2).round(4))])
        po_R_words = pd.concat([po_R_words, pd.DataFrame(np.sum(Rx, axis=1).round(4))])
        po_words = pd.concat([po_words,pd.DataFrame(index_to_word(target_data))])
        label.append(target_class)
    except Exception: 
        pass

positive_lrp = pd.DataFrame({'words': po_words.iloc[:,0], 'sa':po_R_words_SA.iloc[:,0], 'lrp_rele':po_R_words.iloc[:,0]})
positive_lrp.to_csv('positive_lrp.csv')

### 부정리뷰 분석
ne_R_words_SA= pd.DataFrame()
ne_R_words= pd.DataFrame()
ne_words = pd.DataFrame()

negative_list['review_text']=negative_list['review_text'].apply(lambda x: process_text(str(x)))

for i in tqdm(range(len(negative_list))):
    try:        
        list_tokenized_ex = tokenizer3.texts_to_sequences(negative_list['review_text'].iloc[i])
        target_full_data = pad_sequences(list_tokenized_ex, maxlen=max_review_length)
        
        target_data = target_full_data[target_full_data != 0]
        target_class = np.argmax(negative_list['label'].iloc[i])
        
        score, x, Gx, Rx, R_rest = lrp.run(target_data, target_class)    
        
        ne_R_words_SA=pd.concat([ne_R_words_SA, pd.DataFrame(((np.linalg.norm(Gx, ord=2, axis=1))**2).round(4))])
        ne_R_words = pd.concat([ne_R_words, pd.DataFrame(np.sum(Rx, axis=1).round(4))])
        ne_words = pd.concat([ne_words,pd.DataFrame(index_to_word(target_data))])
    except Exception: 
        pass

negative_lrp = pd.DataFrame({'words': ne_words.iloc[:,0], 'negative_sa':ne_R_words_SA.iloc[:,0], 'lrp_rele':ne_R_words.iloc[:,0]})
negative_lrp.to_csv('negative_lrp.csv')


#############################LIME###################################
def predict_proba(arr): ##predict probability 측정
    processed=[]
    for i in arr:
        processed.append(process_text(i))  
    list_tokenized_ex = tokenizer3.texts_to_sequences(processed)
    Ex = pad_sequences(list_tokenized_ex, maxlen=max_review_length)
    pred=loaded_model.predict(Ex)
    returnable=[]
    for i in pred:
        temp=i[1]
        returnable.append(np.array([1-temp,temp])) #I would recommend rounding temp and 1-temp off to 2 places
    return np.array(returnable)

#LIME 모델 설정
class_names=['negative','positive']
explainer= LimeTextExplainer(class_names=class_names)

test_reveiw_list = np_list['review_text'].tolist()                

result_df = pd.DataFrame()

for i in tqdm(test_reveiw_list):
    exp = explainer.explain_instance(i, predict_proba)
    exp_df = pd.DataFrame(exp.as_list())
    result_df = pd.concat([result_df, exp_df], ignore_index=True).round(3)

result_df.columns = ['word', 'value']

# LIME score 0이상 긍정
result_df_p = pd.DataFrame(result_df[result_df['value']>0])
sum_p = result_df_p['word'].value_counts()

##각 단어별 평균계산
sum_p = pd.concat([result_df_p['value'].groupby(result_df_p['word']).mean(),result_df_p['word'].value_counts()], axis=1)
sum_p.to_csv('라임_긍정_가중치.csv')

# LIME score 0이하 부정
result_df_n = result_df[result_df['value']<0]
sum_n = result_df_n['word'].value_counts()

## 각 단어별 평균계산
sum_n = pd.concat([result_df_n['value'].groupby(result_df_n['word']).mean(),result_df_n['word'].value_counts()], axis=1)
sum_n.to_csv('라임_부정_가중치.csv')



#########################anchor##################################
from anchor import anchor_text
import en_core_web_sm
import spacy
import transformers
import torch

#anchor 모델 설정
nlp = spacy.load('en_core_web_sm')

explainer = anchor_text.AnchorText(nlp, class_names=class_names, use_unk_distribution=True) 

def predict_proba_anchor(arr):    ##anchor에 맞는 predict_probability계산
    list_tokenized_ex = tokenizer3.texts_to_sequences([arr])
    Ex = pad_sequences(list_tokenized_ex, maxlen=max_review_length)
    pred=np.argmax(model.predict(Ex))
    return [pred]

#anchor 실험
anchor_df = pd.DataFrame()
for i in range(0,1000):
    t = np_list['review_text'].iloc[i : i+1].values[0]
    exp = explainer.explain_instance(t, predict_proba_anchor, threshold=0.90)    
    anchor_df = pd.concat([anchor_df, pd.DataFrame(exp.exp_map)])

#결과 저장
anchor_df.to_csv('anchor_result.csv')
