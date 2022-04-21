# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:07:58 2022

@author: young
"""
###전체 리뷰 분석
import pandas as pd
import numpy as np
import re
import re
import os
from tqdm import tqdm
import string

import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from textblob import TextBlob, Word
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer

from collections import Counter

import matplotlib.pyplot as plt

import imblearn 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, classification_report

import tensorflow as tf
import shap

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten, SimpleRNN
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

from lime.lime_text import LimeTextExplainer

from collections import Counter
from wordcloud import WordCloud

# path_dir = 'C:/Users/young/OneDrive/바탕 화면/졸업논문/data/action/master'
# file_list = os.listdir(path_dir)

# data = pd.DataFrame()
# for a in file_list:
#     sample = pd.read_csv("C:/Users/young/OneDrive/바탕 화면/졸업논문/data/action/master/"+a, index_col = 0)
#     data = pd.concat([data, sample], ignore_index=True)

####################################
def decontract(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

word_list=['game', 'games', 'time', 'fun', 'hours', 'series', 'issues', 'lot',
           'hunt', 'gameplay', 'get', 'players', 'player', 'capcom', 'times', 'love', 'everything', 'review', 'grind',
           'something', 'bit', 'use', 'nothing', 'money', 'end', 'everyone', 'part', 'point', 'issue', 'fan', 'steam', 'release',
           'lots', 'play', 'thing', 'port', 'yes','mhw', 'iceborn', 'souls', 'anyone', 'enjoy', 'start', 'review', 'buy',
           'rank', 'anything', 'look', 'someone', 'day', 'hate', 'gud', 'need', 'cons', 'recommend', 'titles', 'title', 'year',
           'hunters', 'reviews', 'tons', 'dont', 'hour', 'edit', 'fact', 'ones', 'months', 'days', 'rise', 'ass', 'shit',
           'type', 'see', 'pros', 'areas', 'beat', 'mind', 'guess', 'dark', 'please', 'awesome', 'psp', 'hope', 'thats', 'entry',
           'yeah', 'half', 'xbox', 'lol', 'number', 'grindy', 'cant', 'choice', 'cause', 'alot', 'line', 'room', 'purchase', 'gon',
           'clunky', 'fuck', 'sucks', 'suck', 'damn', 'ton', 'weeks', 'couple', 'week', 'devs', 'hub', 'trash', 'buying', 'pay',
           'value', 'bonk', 'second', 'seconds', 'case', 'rng', 'understand', 'plenty', 'good','tho', 'turn', 'spent', 'elder', 'want',
           'put', 'denuvo', 'thank', 'grinding', 'improvement', 'performance', 'near','need', 'needs', 'waste','january', 'february',
           'march','april','may','june', 'july','august', 'september', 'october', 'november', 'december', 'aaa',
           'abunch', 'accept', 'account', 'addict', 'addicting', 'addiction', 'aight', 'aku', 'allow', 'allows', 'amd', 'anyways',
           'appreciate', 'asik','bagi', 'bem', 'berada', 'brr','brrr','che','con', 'dat','fxxking','god', 'gong', 'goodgame', 'haha',
           'hehe', 'hes','hnter','https', 'jadi', 'jason','lil','pcmasterracecons', 'whatever', 'lasts','beside','euros','bdo','handful',
           'embodies','dogshit', 'refuses','ffxiv','vanilla', 'tea','disappointment','angry','wird','funtime','bother', 'becomes','rip',
           'boring','borning','subpar','gama','taht','lfpg', 'complete','cans','ruins','sides','dumb','drm','headache', 'thousand', 'limp',
           'turbo','swaxe','dood','esteem','safi','month','muh', 'miles','pathetic','dull','emos','tera','bound', 'champ','birth','pieces',
           'answers','silky','meant','poop','ruin','gui','pickles','pog','tranq','keren','fav','tab','max', 'dog', 'shit', 'shxx']

stop_words = set(stopwords.words('english')+word_list) 
shortword = re.compile(r'\W*\b\w{1,2}\b')

def lem(a):   
    sent = TextBlob(a)
    result = " ".join([w.lemmatize() for w in sent.words])
    return result   
    
def process_text(a):
    a = str(re.sub("\S*\d\S*", "", a).strip()) 
    a = a.lower()
    text=decontract(a)    
    a = re.sub(r"[^a-zA-Z0-9]"," ",a) #특수문자 제거
    a = re.sub("([^\x00-\x7F])+","",a) #영어이외제거    
    a = re.sub("(.)\\1{3,}", "\\1", a) #긴 반복문 제거
    a = shortword.sub(' ', a)    #짧은 단어 제거        
        
    tokenizer = TreebankWordTokenizer()
    # tokenize texts
    tokens = tokenizer.tokenize(a)
    
    
    result = []
    for b in tokens:
        b = shortword.sub(' ', b)    #짧은 단어 제거  
        if b not in stop_words:  #stopwords 제거
            result.append(b)   
    
    return result



########################################################

data1 = pd.read_csv("./mh.csv", index_col = 0)
data2 = pd.read_csv("./mh_rise.csv", index_col = 0)
data3 = pd.read_csv("./mh_neg.csv", index_col = 0)
data4 = pd.read_csv("./mh_neg2.csv", index_col = 0)
data5 = pd.read_csv("./mh_neg3.csv", index_col = 0)
data6 = pd.read_csv("./mh_neg4.csv", index_col = 0)
data7 = pd.read_csv("./mh_neg5.csv", index_col = 0)
# data1['label'] = data1['voted_up'].replace({True: 1, False: 0})
# data2['label'] = data2['voted_up'].replace({True: 1, False: 0})
# data1 = data1.dropna(subset=['review'])
# data2 = data2.dropna(subset=['review'])

# data1['label'].value_counts()
# data2['label'].value_counts()

data = pd.concat([data1, data2], ignore_index=True)

data = data.dropna(subset=['review'])

#
data['label'] = data['voted_up'].replace({True: 1, False: 0})

data['label'].value_counts()

#단어 전처리
pre_data = pd.DataFrame(columns =['review_text', 'label'])
pre_data['review_text']=data['review']
pre_data['label'] = data['label']

neg_data1 =pd.DataFrame(columns =['review_text', 'label'])
neg_data1['review_text'] = data3['review']
neg_data1['label'] = data3['score'].replace({"Not Recommended": 0})

neg_data2 =pd.DataFrame(columns =['review_text', 'label'])
neg_data2['review_text'] = data4['review']
neg_data2['label'] = data4['score'].replace({"Not Recommended": 0})

neg_data3 =pd.DataFrame(columns =['review_text', 'label'])
neg_data3['review_text'] = data5['review']
neg_data3['label'] = data5['score'].replace({"Not Recommended": 0})

neg_data4 =pd.DataFrame(columns =['review_text', 'label'])
neg_data4['review_text'] = data6['review']
neg_data4['label'] = data6['score'].replace({"Not Recommended": 0})


pre_data = pd.concat([neg_data2, pre_data, neg_data1, neg_data3, neg_data4], ignore_index=True)
pre_data['review_text'] = pre_data['review_text'].str.lower()
pre_data['review_text']=pre_data['review_text'].apply(lambda x: lem(str(x)))
pre_data['review_text']=pre_data['review_text'].apply(lambda x: process_text(str(x)))
pre_data['label'].value_counts()


tokens_pos = []
for i in tqdm(pre_data['review_text']):
    tokens_pos.append(nltk.pos_tag(i))

detoken=[]

for i in range(0, len(tokens_pos)):
    NN_words = []
    for word, pos in tokens_pos[i]:
        if 'NN' in pos:
            NN_words.append(word)
    detoken.append(TreebankWordDetokenizer().detokenize(NN_words))

pre_data['review_text'] = pd.Series(detoken)
pre_data['review_text'] = pre_data['review_text'].astype('str')

pre_data = pre_data[pre_data['review_text'].map(len)>0]

#단어 빈도 세기
negative_text =pre_data[pre_data['label']==0].review_text
positive_text =pre_data[pre_data['label']==1].review_text


# Initializing Dictionary
d = {}

# counting number of times each word comes up in list of words (in dictionary)

for word in word_list:
    for word in word:
        d[word] = d.get(word, 0) + 1
    
word_freq = []
for key, value in d.items():
    word_freq.append((value, key))    

word_freq.sort(reverse=True) 

pre_data = pre_data.dropna()


len_list=[]
for sentence in pre_data['review_text']:
    len_list.append(len(sentence))

pre_data['text_length'] = len_list

pre_data = pre_data[pre_data['text_length']>=8]

pre_data['label'].value_counts()


#train test set 분리
X_train, X_test, y_train, y_test = train_test_split(pre_data['review_text'], pre_data['label'], test_size=0.3) 

#############LDA - topic modeling

import gensim.corpora as corpora
import gensim

# tokenize texts

tokens = pre_data['review_text'].apply(lambda x: TreebankWordTokenizer().tokenize(str(x)))

dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(text) for text in tokens]


NUM_TOPICS = 5 # 5개의 토픽, k=5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
topics = ldamodel.print_topics()

topics = pd.DataFrame(topics)

topics.to_csv('topic_modeling.csv')
##############################################################################


##토크나이저 
tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(X_train)


#####10회미만 빈도 
threshold = 11
total_cnt = len(tokenizer2.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer2.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

#등장빈도 10회 이하인 단어 제거 
vocab_size = total_cnt - rare_cnt + 1

tokenizer3 = Tokenizer(num_words= vocab_size)
tokenizer3.fit_on_texts(X_train)

list_tokenized_train = tokenizer3.texts_to_sequences(X_train)


#max_review_length 정하기 
print('리뷰의 최대 길이 :',max(len(review) for review in list_tokenized_train))
print('리뷰의 평균 길이 :',sum(map(len, list_tokenized_train))/len(list_tokenized_train))
plt.hist([len(review) for review in list_tokenized_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(min_review_length, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) >= min_review_length):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이상인 샘플의 비율: %s'%(min_review_length, (count / len(nested_list))*100))

min_review_length =8
below_threshold_len(min_review_length, X_train)

max_review_length = 240
X_train = pad_sequences(list_tokenized_train, maxlen=max_review_length)

y_train.value_counts()

############ X _Test
X_test = tokenizer3.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_review_length)

words_list = tokenizer3.index_word

###LSTM 모델 구축
embedding_dim = 100
hidden_units = 128

tf.compat.v1.enable_v2_behavior()
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=128, validation_split=0.3)

loaded_model = load_model('best_model.h5')
loaded_model.summary()

prediction = loaded_model.predict(X_test)
y_pred = (prediction > 0.5)
print("Accuracy of the LSTM model : ", accuracy_score(y_pred, y_test))
print('F1-score: ', f1_score(y_pred, y_test))
print('Confusion matrix:')
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))


###Bi LSTM

model2 = Sequential()
model2.add(Embedding(vocab_size, embedding_dim))
model2.add(Bidirectional(LSTM(hidden_units))) # Bidirectional LSTM을 사용
model2.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history2 = model2.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=128, validation_split=0.2)

loaded_model2 = load_model('best_model.h5')

prediction = loaded_model2.predict(X_test)
y_pred = (prediction > 0.5)
print("Accuracy of the Bi-LSTM model : ", accuracy_score(y_pred, y_test))
print('F1-score: ', f1_score(y_pred, y_test))
print('Confusion matrix:')
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))



#### GRU

model3 = Sequential()
model3.add(Embedding(vocab_size, embedding_dim))
model3.add(GRU(hidden_units)) # GRU을 사용
model3.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history3 = model3.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=128, validation_split=0.3)

loaded_model3 = load_model('best_model.h5')
print("테스트 정확도: %.4f" % (loaded_model3.evaluate(X_test, y_test)[1]))

prediction = loaded_model3.predict(X_test)
y_pred = (prediction > 0.5)
print("Accuracy of the GRU model : ", accuracy_score(y_pred, y_test))
print('F1-score: ', f1_score(y_pred, y_test))
print('Confusion matrix:')
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))


############LIME
import numpy as np

def predict_proba(arr):
    processed=[]
    for i in arr:
        processed.append(process_text(i))  
    list_tokenized_ex = tokenizer3.texts_to_sequences(processed)
    Ex = pad_sequences(list_tokenized_ex, maxlen=max_review_length)
    pred=loaded_model.predict(Ex)
    returnable=[]
    for i in pred:
        temp=i[0]
        returnable.append(np.array([1-temp,temp])) #I would recommend rounding temp and 1-temp off to 2 places
    return np.array(returnable)

result_df = pd.DataFrame()

class_names=['negative','positive']
explainer= LimeTextExplainer(class_names=class_names)

np_list = np.random.choice(pre_data['review_text'], size = 1000, replace=False)    

test_reveiw_list = np_list.tolist()                
for i in tqdm(test_reveiw_list):
    exp = explainer.explain_instance(i, predict_proba)
    exp_df = pd.DataFrame(exp.as_list())
    result_df = pd.concat([result_df, exp_df], ignore_index=True).round(3)

result_df.columns = ['word', 'value']

result_df_p = pd.DataFrame(result_df[result_df['value']>0])
result_df_p['word'].value_counts()

sum_p = result_df_p['value'].groupby(result_df_p['word']).sum()
sum_p_df = pd.DataFrame({'word': sum_p.index.to_list(), 'value':sum_p.values.tolist()})

sum_p_df['normalization'] = (sum_p_df['value']-sum_p_df['value'].mean())/sum_p_df['value'].std()

# mean_p_df_top = mean_p_df[mean_p_df['value']>0.1]

sum_p_df.to_csv("전체리뷰 긍정 가중치_합_정규화.csv")

result_df_n = result_df[result_df['value']<0]
sum_n = result_df_n['value'].groupby(result_df_n['word']).sum()
sum_n_df = pd.DataFrame({'word': sum_n.index.to_list(), 'value':sum_n.values.tolist()})
sum_n_df['normalization'] = (sum_n_df['value']-sum_n_df['value'].mean())/sum_n_df['value'].std()

sum_n_df.to_csv("전체리뷰 부정 가중치_합.csv")


#긍정 top 단어가 들어간 리뷰 수
mean_p_df = pd.read_csv("C:/Users/young/OneDrive/바탕 화면/졸업논문/전체리뷰 긍정 가중치_합.csv")

mean_p_top = mean_p_df[mean_p_df['value']>0.2]

matching1 = [s for s in positive_text if any(xs in s for xs in mean_p_top['word'])]

#부정 top 단어가 들어간 리뷰 수
mean_n_df = pd.read_csv("C:/Users/young/OneDrive/바탕 화면/졸업논문/전체리뷰 부정 가중치_합.csv")
mean_n_top = mean_n_df[mean_n_df['value']< -0.3]

matching2 = [s for s in negative_text if any(xs in s for xs in mean_n_top['word'])]

#긍정 단어 tdm
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=100, stop_words=stop_words, min_df=10)
tdm1 = cv.fit_transform(matching1)
words1 = cv.get_feature_names()

count_mat1 = tdm1.sum(axis=0)
count_mat1 = np.squeeze(np.array(count_mat1))
word_count1 = list(zip(words1, count_mat1))
word_count1 = sorted(word_count1, key=lambda x: x[1], reverse=True)


word_corr = np.corrcoef(tdm1.todense(), rowvar=0)
word_corr_df = pd.DataFrame(word_corr) 
word_corr_df.index =words1
word_corr_df.columns = words1

word_corr_df.to_csv('긍정단어network2.csv')

#부정단어 tdm
cv = CountVectorizer(max_features=100, stop_words=stop_words, min_df=10)
tdm2 = cv.fit_transform(matching2)
words2 = cv.get_feature_names()

count_mat2 = tdm2.sum(axis=0)
count_mat2 = np.squeeze(np.array(count_mat2))
word_count2 = list(zip(words2, count_mat2))
word_count2 = sorted(word_count2, key=lambda x: x[1], reverse=True)



word_corr = np.corrcoef(tdm2.todense(), rowvar=0)
word_corr_df = pd.DataFrame(word_corr) 
word_corr_df.index =words2
word_corr_df.columns = words2

word_corr_df.to_csv('부정단어network2.csv')

###SHAP #####오류 발생으로 오류 해결 중
pip install shap --upgrade
import shap

np_list_seq = tokenizer3.texts_to_sequences(np_list)
np_list_seq = pad_sequences(np_list_seq, maxlen=max_review_length)

explainer = shap.DeepExplainer(model, X_train[:100])

shap_vals = explainer.shap_values(np_list_seq_df[:2].values)


words = tokenizer3.word_index
word_lookup = list()
for i in words.keys():
  word_lookup.append(i)
word_lookup = [''] + word_lookup
shap.summary_plot(shap_vals, feature_names=word_lookup, class_names=tag_encoder.classes_)



