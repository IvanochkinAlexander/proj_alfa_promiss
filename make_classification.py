#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pandas as pd
import re
import os
import sys
from string import punctuation
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.text import TextCollection
from nltk.corpus import stopwords
from nltk.text import TextCollection
from nltk.util import ngrams
import pymorphy2
from pymorphy2 import MorphAnalyzer
import artm
import gensim
from gensim import corpora, models, similarities
from gensim import corpora, models
from gensim.models import Word2Vec
from gensim.models import word2vec
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from catboost import Pool, CatBoostClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from nltk.stem.snowball import RussianStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.metrics import accuracy_score
import heapq
from sklearn.neural_network import multilayer_perceptron
from itertools import combinations_with_replacement
import datetime



def read_data(train_path, test_path):

    """Reading data, concating train and test"""

    #train_dataframe = pd.read_excel(train_path)
    #test_dataframe = pd.read_excel(test_path)

    train_dataframe = pd.read_json(train_path)
    test_dataframe = pd.read_json(test_path)
    train_dataframe = train_dataframe.fillna('isnun')
    test_dataframe = test_dataframe[[u'Суть обращения']].fillna('isnun')
    test_dataframe[u'Процесс'] = np.nan
    train_dataframe['train_test'] = 'train'
    test_dataframe['train_test'] = 'test'
    train_test = pd.concat([train_dataframe, test_dataframe])
    return train_test

def read_data_test(test_path):

    """Source test for submisiion"""

    #test_dataframe = pd.read_excel(test_path)
    test_dataframe = pd.read_json(test_path)
    test_dataframe = test_dataframe[[u'Суть обращения']]
    return test_dataframe

def read_data_for_validation(train_path, n):

    """Reading data, concating train and test"""

    #train_dataframe = pd.read_excel(train_path)
    train_dataframe = pd.read_json(train_path)
    test_dataframe = train_dataframe[n:]
    train_dataframe = train_dataframe[:n]
    y_true = test_dataframe[[u'Суть обращения', u'Процесс']]
    test_dataframe[u'Процесс'] = np.nan
    train_dataframe['train_test'] = 'train'
    test_dataframe['train_test'] = 'test'
    train_test = pd.concat([train_dataframe, test_dataframe])
    return train_test, y_true

def basic_cleaning2(string):

    """Cleaning text from numbers and punctuation"""

    string = string.lower()
    string = re.sub('[0-9\(\)\!\^\%\$\'\"\.;,-\?\{\}\[\]\\/]', ' ', string)
    string = re.sub(' +', ' ', string)
    return string

def preprocess_data (train_test):

    """Replacing  abbreviations and general preprocessing"""

    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(basic_cleaning2)
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub('_', ' ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' п.п. ', u' платежные поручения ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ср-в ', u' средства ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ср-ва ', u' средства ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ден. ', u' денежные ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' р/с ', u' расчетный счет ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ПУ. ', u' пакет услуг ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ДО. ', u' дополнительный офис ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' албо ', u' альфа бизнес онлайн ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' п.п ', u' платежные поручения ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' АЛБО ', u' альфа бизнес онлайн ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' юр. ', u' юридический ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' юл. ', u' юридическое лицо ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ЮЛ. ', u' юридическое лицо ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ЮЛ ', u' юридическое лицо ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' юл ', u' юридическое лицо ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ул. ', u' улица ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' руб. ', u' рубль ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ст. ', u' статья ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ип. ', u' индивидуальный предприниматель ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ген. ', u' генеральный ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' г. ', u' год' , text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' т.к ', u' так как ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' оао ', u' открытое акционерное общество ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' фл. ', u' физическое лицо ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' д/с ', u' денежные средства ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' цб. ', u' центральный банк ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' руб ', u' рубль ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ооо ', u' общество с ограниченной ответственностью ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' юр ', u' юридический ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ао ', u' акционерное общество ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ип ', u' индивидуальный предприниматель ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' «альф ', u' «альф ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' рко ', u' расчетно кассовое обслуживание ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' банк» ', u' банк ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' пу ', u' пакет услуг ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ден ', u' денежный ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' уск ', u' управление по сохранению клиентов ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' цпк ', u' центр поддержки клиентов ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' рф ', u' российская федерация ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ген ', u' генеральный ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' ст ', u' статья ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' фз ', u' федеральный закон ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' зп ', u' зарплатный ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' дс ', u' денежное средство ', text))
    train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u' юла ', u' юридический ', text))
#     train_test[u'Суть обращения'] = train_test[u'Суть обращения'].apply(lambda text : re.sub(u'', u'none', text))
    train_test = train_test.fillna(u'пусто')
    train_test = train_test.reset_index(drop=True)
    train_test = train_test[u'Суть обращения']
    russian_stops = stopwords.words('russian')+\
    [u'это', u'иза', u'свой',u'млрд', u'млн',u'млна',u'тыс',\
     u'трлн', u'вопрос', u'весь', u'который', u'наш', '-', ',',\
     u'это', u'вопрос', u'весь', u'самый', u'ваш', u'наш', u'почему', u'свой',\
     '=', '{', '+', '}', 'var', u'–', '1', 'if', '/', '5', u'г.', '});', '0;', 'return', 'i', '>', 'listid', 'isfavorite',\
     'false;', 'webid', 'result', 'function(data)', '2', '3', '4', '5', '6', '7', '8', '9', 'url', u'(в',\
     'function', '000', 'window.tops.constants.contenttypes.akbusermanual', U'(переть', u'случае,', '10', '12',\
     u'ucs.ести', u'«мыть', u'(ить', u'(переть', u'(ести', u'существующих)*ести', u'(возникать', u'(мочь']+\
    ['*', u'г', u'№', u'р', 'ot', 'n', u'a', 'al', 'fa',  u'ещё']+\
    [u'аба',u'т.',u'к.',u'г.', u'александровна',u'сергеевна',u'владимировна',u'елена',u'екатерина',u'ольга',u'николаевна',u'юлия',u'татьяна',u'наталья',u'викторовна',u'анна',u'ирина',u'анастасия',u'юрьевна',u'александрович',u'владимирович',u'александр',u'светлана',u'сергеевич',u'андреевна',u'анатольевна',u'мария',u'валерьевна',u'сергей',u'михайловна',u'марина',u'игоревна',u'алексеевна',u'дмитрий',u'алексей',u'евгеньевна',u'олеговна',u'николаевич',u'андрей',u'дарья',u'евгения',u'васильевна',u'виктория',u'ксения',u'викторович',u'вячеславовна',u'геннадьевна',u'юрьевич',u'ивановна',u'оксана',u'кристина',u'надежда',u'владимир',u'анатольевич',u'александра',u'евгений',u'дмитриевна',u'андреевич',u'павловна',u'валерьевич',u'михайлович',u'петровна',u'алексеевич',u'людмила',u'михаил',u'максим',u'алина',u'олеся',u'денис',u'игоревич',u'евгеньевич',u'павел',u'яна',u'васильевич',u'игорь',u'леонидовна',u'наталия',u'константиновна',u'роман',u'галина',u'борисовна',u'витальевна',u'антон',u'алена',u'олегович',u'олег',u'николай',u'иванович',u'иван',u'лилия',u'юрий',u'алёна',u'валентина',u'инна',u'вадимовна',u'маргарита',u'дмитриевич',u'любовь',u'иванова',u'вячеславович',u'валерия',u'геннадьевич',u'константин',u'лариса',u'кузнецова',u'артем',u'илья',u'виталий',u'виктор',u'вера',u'диана',u'борисович',u'валентиновна',u'кирилл',u'петрович',u'вячеслав',u'павлович',u'григорьевна',u'елизавета',u'попова',u'полина',u'нина',u'альбина',u'вероника',u'владислав',u'вадим',u'станиславовна',u'эдуардовна',u'никита',u'эльвира',u'георгиевна',u'владиславовна',u'леонидович',u'валерий',u'руслан',u'регина',u'витальевич',u'василий',u'смирнова',u'константинович',u'васильева',u'петрова',u'алла',u'новикова',u'вадимович',u'федоровна',u'анатолий',u'карина',u'семенова',u'станислав',u'иванов',u'алексеева',u'павлова',u'морозова',u'степанова',u'романовна',u'егорова',u'волкова',u'макарова',u'валериевна',u'жанна',u'софья',u'лидия',u'орлова',u'динара',u'валентинович',u'маратовна',u'альбертовна',u'николаева',u'анжелика',u'федорова',u'кузнецов',u'сергеева',u'андреева',u'сорокина',u'романова',u'козлова',u'попов',u'ангелина',u'тамара',u'артур',u'дина',u'яковлева',u'григорьева',u'артём',u'григорьевич',u'аркадьевна',u'смирнов',u'бондаренко',u'максимова',u'антонина',u'алиса',u'руслановна',u'никитина',u'алсу',u'анжела',u'лебедева',u'михайлова',u'полякова',u'эльмира',u'��лия',u'радиковна',u'воробьева',u'матвеева',u'сидорова',u'захарова',u'зайцева',u'миронова',u'егор',u'соколова',u'кузьмина',u'федорович',u'гузель',u'алевтина',u'ильдаровна',u'петров',u'шевченко',u'гаврилова',u'мельникова',u'георгий',u'гульнара',u'георгиевич',u'рамилевна',u'фролова',u'ковалева',u'ильина',u'геннадий',u'леонид',u'петр',u'беляева',u'мартынова',u'соловьева',u'бойко',u'борисова',u'карпова',u'борис',u'альфия',u'ринатовна',u'ким', u'юриевич',u'евгениевич', u'василиевич', u'анатолиевич']+\
    [u'сергеевич',u'л',u'елена',u'ольга',u'александрович',u'владимирович',u'инна']
    russian_stops = list(stopwords.words('russian'))
    russian_stops.pop(russian_stops.index(u'не'))
    russian_stops.extend([u'аба',u'т.',u'к.',u'г.', u'александровна',u'сергеевна',u'владимировна',u'елена',u'екатерина',u'ольга',u'николаевна',u'юлия',u'татьяна',u'наталья',u'викторовна',u'анна',u'ирина',u'анастасия',u'юрьевна',u'александрович',u'владимирович',u'александр',u'светлана',u'сергеевич',u'андреевна',u'анатольевна',u'мария',u'валерьевна',u'сергей',u'михайловна',u'марина',u'игоревна',u'алексеевна',u'дмитрий',u'алексей',u'евгеньевна',u'олеговна',u'николаевич',u'андрей',u'дарья',u'евгения',u'васильевна',u'виктория',u'ксения',u'викторович',u'вячеславовна',u'геннадьевна',u'юрьевич',u'ивановна',u'оксана',u'кристина',u'надежда',u'владимир',u'анатольевич',u'александра',u'евгений',u'дмитриевна',u'андреевич',u'павловна',u'валерьевич',u'михайлович',u'петровна',u'алексеевич',u'людмила',u'михаил',u'максим',u'алина',u'олеся',u'денис',u'игоревич',u'евгеньевич',u'павел',u'яна',u'васильевич',u'игорь',u'леонидовна',u'наталия',u'константиновна',u'роман',u'галина',u'борисовна',u'витальевна',u'антон',u'алена',u'олегович',u'олег',u'николай',u'иванович',u'иван',u'лилия',u'юрий',u'алёна',u'валентина',u'инна',u'вадимовна',u'маргарита',u'дмитриевич',u'любовь',u'иванова',u'вячеславович',u'валерия',u'геннадьевич',u'константин',u'лариса',u'кузнецова',u'артем',u'илья',u'виталий',u'виктор',u'вера',u'диана',u'борисович',u'валентиновна',u'кирилл',u'петрович',u'вячеслав',u'павлович',u'григорьевна',u'елизавета',u'попова',u'полина',u'нина',u'альбина',u'вероника',u'владислав',u'вадим',u'станиславовна',u'эдуардовна',u'никита',u'эльвира',u'георгиевна',u'владиславовна',u'леонидович',u'валерий',u'руслан',u'регина',u'витальевич',u'василий',u'смирнова',u'константинович',u'васильева',u'петрова',u'алла',u'новикова',u'вадимович',u'федоровна',u'анатолий',u'карина',u'семенова',u'станислав',u'иванов',u'алексеева',u'павлова',u'морозова',u'степанова',u'романовна',u'егорова',u'волкова',u'макарова',u'валериевна',u'жанна',u'софья',u'лидия',u'орлова',u'динара',u'валентинович',u'маратовна',u'альбертовна',u'николаева',u'анжелика',u'федорова',u'кузнецов',u'сергеева',u'андреева',u'сорокина',u'романова',u'козлова',u'попов',u'ангелина',u'тамара',u'артур',u'дина',u'яковлева',u'григорьева',u'артём',u'григорьевич',u'аркадьевна',u'смирнов',u'бондаренко',u'максимова',u'антонина',u'алиса',u'руслановна',u'никитина',u'алсу',u'анжела',u'лебедева',u'михайлова',u'полякова',u'эльмира',u'алия',u'радиковна',u'воробьева',u'матвеева',u'сидорова',u'захарова',u'зайцева',u'миронова',u'егор',u'соколова',u'кузьмина',u'федорович',u'гузель',u'алевтина',u'ильдаровна',u'петров',u'шевченко',u'гаврилова',u'мельникова',u'георгий',u'гульнара',u'георгиевич',u'рамилевна',u'фролова',u'ковалева',u'ильина',u'геннадий',u'леонид',u'петр',u'беляева',u'мартынова',u'соловьева',u'бойко',u'борисова',u'карпова',u'борис',u'альфия',u'ринатовна',u'ким', u'юриевич',u'евгениевич', u'василиевич', u'анатолиевич'])# это так pymorphy АБ превращает в нормальную форму
    morph = pymorphy2.MorphAnalyzer()
    collection = [wrk_words_wt_no(text) for text in train_test]

    ###заменим пустые и прочие
    temp = pd.DataFrame(collection)
    temp['len'] = temp[0].apply(len)
    temp['len'].value_counts()
    temp.loc[temp['len']==0, 0] = u'прочее'
    temp[0] = temp[0].fillna(u'прочее')
    collection = temp[0]

    return collection

def No_with_word(token_text):

    """Concating NO with words"""

    tmp=''
    for i,word in enumerate(token_text):
        if word==u'не':
            tmp+=("_".join(token_text[i:i+2]))
            tmp+= ' '
        else:
            if token_text[i-1]!=u'не':
                tmp+=word
                tmp+=' '
    return tmp

def wrk_words_wt_no(sent):

    """Making stemming"""
#     morph = pymorphy2.MorphAnalyzer()
    stemmer = RussianStemmer()
    words=word_tokenize(sent.lower())
    try:
        arr=[]
        for i in range(len(words)):
            if re.search(u'[а-яА-Я]',words[i]):
                arr.append(stemmer.stem(words[i]))###стемминг
#                 arr.append(morph.parse(words[i])[0].normal_form)###лемматизация
        words1=[w for w in arr if w not in russian_stops]
        words1=No_with_word(words1)
        return words1
    except TypeError:
        pass

def make_bow(col):

    """Make bow from train_test"""

    binVectorizer = CountVectorizer(binary=True, ngram_range=(1, 1), min_df = 2, max_df=20000, max_features=20000)
#     binVectorizer = CountVectorizer(binary=True, ngram_range=(1, 1), min_df = 2)

    counts = binVectorizer.fit_transform(np.array(collection))
    bow_df = pd.DataFrame(counts.toarray(), columns=binVectorizer.get_feature_names())
    to_include = []
    for i in bow_df.columns:
        if bow_df[i].sum()>=2:
            to_include.append(i)
    bow_df = bow_df[to_include]
    return bow_df

def review_to_wordlist(review):

    """Convert collection to wordlist"""

    words = review.lower().split()
    words = [w for w in words]
    return(words)

def make_w2v(collection):

    """Make w2v model"""

    collection_splited = pd.DataFrame(collection).apply(lambda x:x.apply(review_to_wordlist))
    model = word2vec.Word2Vec(collection_splited[0], size=300, window=10, workers=4, min_count=2)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    return w2v, model

def make_w2v_mean(collection):

    """Mean transformation"""

    collection_df = pd.DataFrame(collection).apply(lambda x:x.apply(review_to_wordlist))
#     collection_df = pd.DataFrame(collection)
    data_mean=mean_vectorizer(w2v).fit(collection_df[0]).transform(collection_df[0])
    w2v_mean = pd.DataFrame(data_mean).reset_index(drop=True)
    return w2v_mean

def make_w2v_tfidf(collection):

    """Tf-idf transformation"""

    collection_df = pd.DataFrame(collection).apply(lambda x:x.apply(review_to_wordlist))
    data_tfidf=tfidf_vectorizer(w2v).fit(collection_df[0]).transform(collection_df[0])
    w2v_tfidf = pd.DataFrame(data_tfidf).reset_index(drop=True)
    return w2v_tfidf

class mean_vectorizer(object):

    """Preprocess word embedding using mean transformation"""

    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(w2v.values())))

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class tfidf_vectorizer(object):

    """Class to preprocess word embedding using tf-idf"""

    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(w2v.values())))

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

def make_artm(col):

    """Get artm theta matrixes"""

    collection_train = pd.DataFrame(collection).iloc[train_index].reset_index()
    collection_test = pd.DataFrame(collection).iloc[test_index].reset_index()
    le=LabelEncoder()
    y_transormed = le.fit_transform(df_y)

    arr=[]
    for index_number, i, c in zip(collection_train['index'], collection_train[0], y_transormed):
        arr.append(str(index_number) + ' |@default_class ' + unicode(i) + ' |@labels_class ' + unicode(c))

    arr_test=[]
    for index_number, i in zip(collection_test['index'], collection_test[0]):
        arr_test.append(str(index_number) +  ' |@default_class ' + unicode(i))

    pd.DataFrame(arr,index=None).to_csv('leaver_vw_form.txt',sep='\t',encoding='UTF-8',index=False, header=None)
    pd.DataFrame(arr_test,index=None).to_csv('leaver_vw_form_test.txt',sep='\t',encoding='UTF-8',index=False, header=None)

    batch_vectorizer = artm.BatchVectorizer(data_path="leaver_vw_form.txt", data_format="vowpal_wabbit", target_folder="leaver_vw_form_train", batch_size=100)
    batch_vectorizer_test = artm.BatchVectorizer(data_path="leaver_vw_form_test.txt", data_format="vowpal_wabbit", target_folder="leaver_vw_form_test", batch_size=100)

    T = pd.DataFrame(df_y)[u'Процесс'].nunique()
    print ("количество тем составляет - {}".format(T))# количество тем
    topic_names=["sbj"+str(i) for i in range(T)]

    model_artm = artm.ARTM(num_topics=T,
                           topic_names=topic_names,
                           class_ids={'@default_class':1, '@labels_class':700},
                           num_document_passes=10,
                           seed=79,
                           reuse_theta=True,
                           cache_theta=True,
                           scores=[artm.TopTokensScore(name='top_tokens_score', num_tokens=30, class_id='@default_class')],
                           regularizers=[artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=-0.15)])

    dictionary=artm.Dictionary(name='dictionary')
    dictionary.gather(batch_vectorizer.data_path)

    model_artm.initialize('dictionary')

    dictionary.filter(min_tf=2,  min_df_rate=0.01)

    model_artm.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score',class_id='@labels_class'))
    model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_def',class_ids=['@default_class']))
    model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_lab',class_ids=['@labels_class']))

    model_artm.scores.add(artm.PerplexityScore(name='PerplexityScore',dictionary='dictionary'))
    model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)

    test_transformed = model_artm.transform(batch_vectorizer_test, predict_class_id ='@labels_class').T
    train_transformed = model_artm.transform(batch_vectorizer, predict_class_id ='@labels_class').T

    test_transformed = test_transformed.reset_index().sort_values('index')
    test_transformed = test_transformed.reset_index(drop=True)
    del test_transformed['index']
    test_transformed = test_transformed[sorted(test_transformed.columns)]
    train_transformed = train_transformed.reset_index().sort_values('index')
    del train_transformed['index']
    train_transformed = train_transformed[sorted(train_transformed.columns)]
    train_transformed = train_transformed.reset_index(drop=True)
    artm_transformed = pd.concat([train_transformed, test_transformed], axis=0).reset_index(drop=True)

    return artm_transformed

def train_xgboost(train, y, test, proba):

    """Train and predict using xgboost"""

    clf_xgboost = xgboost.XGBClassifier()
    clf_xgboost = clf_xgboost.fit(train, y)
    if proba == 'probability':

        preds = clf_xgboost.predict_proba(test)

        labels2idx = {label: i for i, label in enumerate(clf_xgboost.classes_)}
        preds = np.array(preds)
        sub = pd.DataFrame()
        for label in clf_xgboost.classes_:
            sub[label] = preds[:, labels2idx[label]]
        preds = sub
    else:
        preds = clf_xgboost.predict(test)

    return preds


def train_mlp(train, y, test, proba):

    """Train neural network"""

    clf_mlp = multilayer_perceptron.MLPClassifier(hidden_layer_sizes=(128,64,),
                                        activation='relu',
                                        solver='adam',
                                        alpha=0.0001,
                                        batch_size=64,
                                        learning_rate='constant',
                                        learning_rate_init=0.001,
                                        power_t=0.5,
                                        max_iter=20,
                                        shuffle=True,
                                        random_state=None,
                                        tol=0.0001,
                                        verbose=False,
                                        warm_start=False,
                                        momentum=0.9,
                                        nesterovs_momentum=True,
                                        early_stopping=True,
                                        validation_fraction=0.1,
                                        beta_1=0.9,
                                        beta_2=0.999, epsilon=1e-08)

    clf_mlp = clf_mlp.fit(train, y)
    if proba == 'probability':

        preds = clf_mlp.predict_proba(test)

        labels2idx = {label: i for i, label in enumerate(clf_mlp.classes_)}
        preds = np.array(preds)
        sub = pd.DataFrame()
        for label in clf_mlp.classes_:
            sub[label] = preds[:, labels2idx[label]]
        preds = sub
    else:
        preds = clf_mlp.predict(test)

    return preds

def train_lr(train, y, test, proba):

    """Train and predict using logistic regression"""

    clf_lr = LogisticRegression(solver="liblinear",
                                multi_class="ovr")
    clf_lr = clf_lr.fit(train, y)
    if proba == 'probability':

        preds = clf_lr.predict_proba(test)

        labels2idx = {label: i for i, label in enumerate(clf_lr.classes_)}
        preds = np.array(preds)
        sub = pd.DataFrame()
        for label in clf_lr.classes_:
            sub[label] = preds[:, labels2idx[label]]
        preds = sub

    else:
        preds = clf_lr.predict(test)

    return preds

def make_cross_val(train, y, clf_custom):

    """Show quality on cross-validation"""

    clf = clf_custom
    cv_scores = []
    folds=3
    from sklearn.cross_validation import KFold
    kf = KFold(len(y), n_folds=folds, shuffle=True, random_state=2016)
    for i, (train_, test_) in enumerate(kf):
        dev_X, val_X = train.ix[train_], train.ix[test_]
        dev_y, val_y = y.ix[train_], y.ix[test_]
        preds = clf(dev_X, dev_y, val_X, 'actual')
        cv_scores.append(accuracy_score(preds, val_y))
    # print ('mean quality of{}'.format(clf), (np.mean(cv_scores)))

def mix_predictions_curtesian(preds_all, list_of_indexes):

    """Finding best combination of predictions"""

    ### make cartesian product
    all_list = []
    for i in combinations_with_replacement(list_of_indexes, 2):
        if i[0] !=i[1]:
            all_list.append(list(i))
    for list_of_indexes in all_list:
        for num_val, i in enumerate(list_of_indexes):
            preds_temp = preds_all[preds_all['clf']==i]
            del preds_temp['clf']
            if num_val == 0:
                preds_first=preds_temp.values
            else:
                preds_next = preds_temp.values
                preds_first+=preds_next
        result = pd.DataFrame(preds_first/len(list_of_indexes))
        today_time = datetime.datetime.now()
        result['clf'] = str(today_time.hour) + '_' + str(today_time.minute) + '_' +  str(today_time.second) + '_' +  str(today_time.microsecond)
        result.columns = preds_all.columns
        preds_all = pd.concat([preds_all, result], axis=0)
    return preds_all

def mix_predictions(preds_all, list_of_indexes):

    """Average blending of appropriate predictions"""

    for num_val, i in enumerate(list_of_indexes):
        preds_temp = preds_all[preds_all['clf']==i]
        del preds_temp['clf']
        if num_val == 0:
            preds_first=preds_temp.values
        else:
            preds_next = preds_temp.values
            preds_first+=preds_next
    result = pd.DataFrame(preds_first/len(list_of_indexes))
    today_time = datetime.datetime.now()
    result['clf'] = str(today_time.hour) + '_' + str(today_time.minute) + '_' +  str(today_time.second) + '_' +  str(today_time.microsecond)
    result.columns = preds_all.columns
    preds_all = pd.concat([preds_all, result], axis=0)
    return preds_all

def make_submission (preds_all, source_test):

    """Takes top-3 probabilities and concates with source data"""

    temp_value = preds_all['clf']
    preds_all= preds_all.loc[:,preds_all.columns !='clf']

    max_3_all =[]
    for i in range(len(preds_all)):
        nums = preds_all.iloc[i].values
        max_3 = heapq.nlargest(3, nums)
        max_3_all.append(max_3)

    preds_all = pd.concat([preds_all.reset_index(drop=True), pd.DataFrame(max_3_all)], axis=1)
    cols = preds_all.iloc[:,:preds_all.shape[1]-3].columns
    target_cols=[0,1,2]

    for i in cols:
        for target_col in target_cols:
            preds_all.loc[preds_all[i]==preds_all[target_col], 'max_value{}'.format(target_col)] = i
    preds_all['clf'] = temp_value.values

    for i in preds_all['clf'].unique():
        submission = pd.concat([preds_all[preds_all['clf']==i].reset_index(drop=True), source_test],axis=1)
        try:
            submission['score_true'] = (submission['max_value0']==submission[u'Процесс'])
            submission['score_true'] = submission['score_true'].astype(int)
            score_true = submission['score_true'].sum()
            print (i)
            print (score_true/len(submission))
        except:
            pass
    submission.to_json('../output/promiss/submisssion{}.json'.format(i))
    return submission

def split_test (path_test):

    """If test shape is higher than  60l script splits into 25k batches"""

    test_to_split = pd.read_excel(path_test)
    num_parts = int(round(test_to_split.shape[0]/25000))
    batch_size = int(test_to_split.shape[0]/num_parts)
    test_to_split.shape[0]/7
    ### нарезка файлов
    max_index=0
    for i in range(int(num_parts)):
        part = test_to_split.iloc[max_index:(max_index+batch_size)]
        max_index = part.index.max()+1
        part.to_excel('input/10_albo_test_{}.xlsx'.format(i), index=False)
        print (max_index)

    all_dfs_input = pd.DataFrame()
    for i in range(int(num_parts)):
        temp = pd.read_excel('input/10_albo_test_{}.xlsx'.format(i))
        all_dfs_input = pd.concat([all_dfs_input, temp],axis=0)

    all_dfs_input.to_excel('temp_input3.xlsx', index=False)

import random

def reduce_train(paths_train):

    """If class count is higher than 3000 script cuts all samples above 3000"""

    pre_train = pd.read_excel(path_train)
    pre_train = pre_train.rename(columns={u'Текст письма': u'Суть обращения', u'Процесс 1' : u'Процесс'})
    pre_train[u'Суть обращения'] = pre_train[u'Категория'].astype(unicode).fillna(0).replace('nan', '') + ' ' + pre_train[u'Суть обращения'].astype(unicode)
    pre_train = pre_train[[u'Суть обращения', u'Процесс']]

    cut_df = pd.DataFrame()
    for i in pre_train[u'Процесс'].unique():
        temp = pre_train[pre_train[u'Процесс'] == i]
        if temp.shape[0]>3000:
            temp = temp[:3000]
        else:
            temp = temp
        cut_df = pd.concat([cut_df, temp], axis=0)

    cut_df.to_excel('input/train_albo_v4.xlsx', index=False)

def reduce_train_prob (paths_train):
    all_trains = pd.DataFrame()
    for i in paths_train:
        temp_df = pd.read_excel(i)
        temp_df = temp_df.rename(columns = {u'Категория_CJM':u'Процесс 1', u'Проблематика_Лиза':u'Проблема'})
        temp_df = temp_df[[u'Категория', u'Тема письма', u'Текст письма',u'Процесс 1', u'Проблема']]
        all_trains = pd.concat([all_trains, temp_df],axis=0)


    all_trains[u'Категория'] = all_trains[u'Категория'].apply(lambda x : unicode(x).replace(u'Стандарт', ''))
    all_trains[u'Суть обращения'] = all_trains[u'Категория'].astype(unicode)  + ' ' + all_trains[u'Тема письма'].astype(unicode)+ ' '+ all_trains[u'Текст письма'] .astype(unicode)
    all_trains['count'] = all_trains[u'Суть обращения'].apply(lambda x : len(x))
    all_trains = all_trains[all_trains['count']>15]
    all_trains[u'Процесс'] = all_trains[u'Процесс 1'] + '_' + all_trains[u'Проблема']
    all_trains = all_trains[[u'Суть обращения', u'Процесс']]
    cut_df = pd.DataFrame()
    for i in all_trains[u'Процесс'].unique():
        temp = all_trains[all_trains[u'Процесс'] == i]
        if temp.shape[0]>1500:
            temp = temp.sample(1500)
        elif temp.shape[0]<50:
            temp = pd.DataFrame()
        else:
            temp = temp
        cut_df = pd.concat([cut_df, temp], axis=0)
#         cut_df.to_excel('input/train_albo_v4_prob.xlsx', index=False)
    return cut_df

def delete_operator(path_sent):
    chat = pd.read_excel(path_sent)
    chat[u'Суть обращения'] = chat[u'Суть обращения'].apply(lambda x : x+'>>>')
    chat[u'Суть обращения'] = chat[u'Суть обращения'].apply(lambda x : x.replace('operator:', '<<<'))
    chat[u'Суть обращения'] = chat[u'Суть обращения'].apply(lambda x : x.replace('client:', '>>>'))
    def replace_operator (col):
        return re.sub('<<<[^>]+>>>', '', col)
    chat[u'Суть обращения'] = chat[u'Суть обращения'].apply(replace_operator)
    chat.to_excel('input/10_chat_v2_sent.xlsx')

###temp
# pos = pd.read_csv(u'Архив/positive.csv', sep=';', header=None, encoding='utf-8')
# neg = pd.read_csv(u'Архив/negative.csv', sep=';', header=None,  encoding='utf-8')
# pos_df = pd.DataFrame(pos[3][25000:40000])
# neg_df =  pd.DataFrame(neg[3][25000:40000])
# pos_df[u'Процесс'] = 'pos'
# neg_df[u'Процесс'] = 'neg'
# train_sent = pd.concat([pos_df, neg_df], axis=0)
# train_sent = train_sent.rename(columns={3:u'Суть обращения'})
# train_sent[u'Суть обращения'] = train_sent[u'Суть обращения'].apply(basic_cleaning2)
# train_sent.to_excel('input/train_sent.xlsx', index=False)

# %%time
### для больших файлов из интернет банка необходимо разбить тест на части и уменьшить трейн
# paths_train = [u'//dprcdbpt/UOIP_FILES/Архив БД/Dashboard_source/IB/IB_2017_09.xlsx',\
#                u'//dprcdbpt/UOIP_FILES/Архив БД/Dashboard_source/IB/IB_2017_08.xlsx',]
# paths_train =  ['input/06_07_09_train.xlsx',]
# path_test = 'input/10_site.xlsx'

# split_test(path_test)
# reduce_train(path_train)

# %%time
# cut_df = reduce_train_prob (paths_train)

# print cut_df.shape

# path_sent = 'input/10_chat_v2.xlsx'
# delete_operator(path_sent)

# for i in range(int(num_parts)):

# for i in range(2):
#     print i

    # i='10_chat.xlsx'
#     train_path = 'input/train_albo_v4.xlsx'
#     test_path =  'input/10_albo_test_{}.xlsx'.format(i)

train_path = '../input/06_07_09_train.json'
# train_path = 'input/train_sent.xlsx'
test_path = '../output/promiss/test.json'

target = u'Процесс'
# train_test, y_true = read_data_for_validation(train_path, 2500)###function for validation
train_test = read_data(train_path, test_path)#function for general train and test

train_test = train_test.reset_index(drop=True)
train_index = train_test[train_test['train_test']=='train'].index
test_index = train_test[train_test['train_test']=='test'].index
df_y = train_test.iloc[train_index][target]
# russian_stops = list(stopwords.words('russian'))

russian_stops = stopwords.words('russian')+\
[u'это', u'иза', u'свой',u'млрд', u'млн',u'млна',u'тыс',\
 u'трлн', u'вопрос', u'весь', u'который', u'наш', '-', ',',\
 u'это', u'вопрос', u'весь', u'самый', u'ваш', u'наш', u'почему', u'свой',\
 '=', '{', '+', '}', 'var', u'–', '1', 'if', '/', '5', u'г.', '});', '0;', 'return', 'i', '>', 'listid', 'isfavorite',\
 'false;', 'webid', 'result', 'function(data)', '2', '3', '4', '5', '6', '7', '8', '9', 'url', u'(в',\
 'function', '000', 'window.tops.constants.contenttypes.akbusermanual', U'(переть', u'случае,', '10', '12',\
 u'ucs.ести', u'«мыть', u'(ить', u'(переть', u'(ести', u'существующих)*ести', u'(возникать', u'(мочь']+\
['*', u'г', u'№', u'р', 'ot', 'n', u'a', 'al', 'fa',  u'ещё']+\
[u'аба',u'т.',u'к.',u'г.', u'александровна',u'сергеевна',u'владимировна',u'елена',u'екатерина',u'ольга',u'николаевна',u'юлия',u'татьяна',u'наталья',u'викторовна',u'анна',u'ирина',u'анастасия',u'юрьевна',u'александрович',u'владимирович',u'александр',u'светлана',u'сергеевич',u'андреевна',u'анатольевна',u'мария',u'валерьевна',u'сергей',u'михайловна',u'марина',u'игоревна',u'алексеевна',u'дмитрий',u'алексей',u'евгеньевна',u'олеговна',u'николаевич',u'андрей',u'дарья',u'евгения',u'васильевна',u'виктория',u'ксения',u'викторович',u'вячеславовна',u'геннадьевна',u'юрьевич',u'ивановна',u'оксана',u'кристина',u'надежда',u'владимир',u'анатольевич',u'александра',u'евгений',u'дмитриевна',u'андреевич',u'павловна',u'валерьевич',u'михайлович',u'петровна',u'алексеевич',u'людмила',u'михаил',u'максим',u'алина',u'олеся',u'денис',u'игоревич',u'евгеньевич',u'павел',u'яна',u'васильевич',u'игорь',u'леонидовна',u'наталия',u'константиновна',u'роман',u'галина',u'борисовна',u'витальевна',u'антон',u'алена',u'олегович',u'олег',u'николай',u'иванович',u'иван',u'лилия',u'юрий',u'алёна',u'валентина',u'инна',u'вадимовна',u'маргарита',u'дмитриевич',u'любовь',u'иванова',u'вячеславович',u'валерия',u'геннадьевич',u'константин',u'лариса',u'кузнецова',u'артем',u'илья',u'виталий',u'виктор',u'вера',u'диана',u'борисович',u'валентиновна',u'кирилл',u'петрович',u'вячеслав',u'павлович',u'григорьевна',u'елизавета',u'попова',u'полина',u'нина',u'альбина',u'вероника',u'владислав',u'вадим',u'станиславовна',u'эдуардовна',u'никита',u'эльвира',u'георгиевна',u'владиславовна',u'леонидович',u'валерий',u'руслан',u'регина',u'витальевич',u'василий',u'смирнова',u'константинович',u'васильева',u'петрова',u'алла',u'новикова',u'вадимович',u'федоровна',u'анатолий',u'карина',u'семенова',u'станислав',u'иванов',u'алексеева',u'павлова',u'морозова',u'степанова',u'романовна',u'егорова',u'волкова',u'макарова',u'валериевна',u'жанна',u'софья',u'лидия',u'орлова',u'динара',u'валентинович',u'маратовна',u'альбертовна',u'николаева',u'анжелика',u'федорова',u'кузнецов',u'сергеева',u'андреева',u'сорокина',u'романова',u'козлова',u'попов',u'ангелина',u'тамара',u'артур',u'дина',u'яковлева',u'григорьева',u'артём',u'григорьевич',u'аркадьевна',u'смирнов',u'бондаренко',u'максимова',u'антонина',u'алиса',u'руслановна',u'никитина',u'алсу',u'анжела',u'лебедева',u'михайлова',u'полякова',u'эльмира',u'алия',u'радиковна',u'воробьева',u'матвеева',u'сидорова',u'захарова',u'зайцева',u'миронова',u'егор',u'соколова',u'кузьмина',u'федорович',u'гузель',u'алевтина',u'ильдаровна',u'петров',u'шевченко',u'гаврилова',u'мельникова',u'георгий',u'гульнара',u'георгиевич',u'рамилевна',u'фролова',u'ковалева',u'ильина',u'геннадий',u'леонид',u'петр',u'беляева',u'мартынова',u'соловьева',u'бойко',u'борисова',u'карпова',u'борис',u'альфия',u'ринатовна',u'ким', u'юриевич',u'евгениевич', u'василиевич', u'анатолиевич']+\
[u'сергеевич',u'л',u'елена',u'ольга',u'александрович',u'владимирович',u'инна']
# russian_stops = list(stopwords.words('russian'))
russian_stops.pop(russian_stops.index(u'не'))

collection = preprocess_data(train_test)
w2v, model = make_w2v(collection)

# %%time
###подготовим датасеты
bow_df = make_bow(collection)

# w2v_mean_df= make_w2v_mean(collection)
w2v_tfidf_df =  make_w2v_tfidf(collection)

# temp = pd.DataFrame(collection)
# temp['len'] = temp[0].apply(len)
# temp['len'].value_counts()
# temp.loc[temp['len']==0, 0] = u'пусто'
# temp[0] = temp[0].fillna(u'пусто')
# collection = temp[0]

artm_df = make_artm(collection)

print (bow_df.shape[0])
# print w2v_mean_df.shape[0]
print (w2v_tfidf_df.shape[0])
print (artm_df.shape[0])

list_clfs = [train_xgboost, train_lr]
data_sources = [bow_df, w2v_tfidf_df, artm_df]

# list_clfs = [train_lr]
# data_sources = [w2v_tfidf_df, artm_df]

#making cross-validation
# for num, source in enumerate(data_sources):
#     print num
#     train = source.iloc[train_index]
#     y = train_test.iloc[train_index][u'Процесс']
#     for i in list_clfs:
#         make_cross_val(train, y, i)

### predicting test values
preds_all=pd.DataFrame()
for num_source, source in enumerate(data_sources):
    print (num_source)
    train = source.iloc[train_index]
    y = train_test.iloc[train_index][u'Процесс']
    test = source.iloc[test_index]
    for num_clf, i in enumerate(list_clfs):
        if (num_source == 0) & (num_clf == 1):
            print ('пропускаем')
            pass
        elif (num_source == 1) & (num_clf == 0):
            print ('пропускаем')
            pass
        elif (num_source == 2) & (num_clf == 0):
            print ('пропускаем')
            pass
        else:
            print (i)
            preds = i(train, y, test, 'probability')
            preds = pd.DataFrame(preds)
            preds['clf']=str(num_source) + ' ' + str(num_clf)
            preds_all = pd.concat([preds_all, preds])

preds_all = mix_predictions(preds_all, ['0 0', '1 1', '2 1'])
# preds_all = mix_predictions(preds_all, ['0 0', '1 0'])
#
source_test=read_data_test(test_path)###for general test
# source_test = y_true.reset_index(drop=True)#for validation

sub = make_submission(preds_all, source_test)
sub = sub[['max_value0']]
#parsed_date_money = pd.read_excel('/root/projects/output/promiss/parsed_date_money.xlsx')
parsed_date_money = pd.read_json('/root/projects/output/promiss/parsed_date_money.json')
parsed_date_money = pd.concat([parsed_date_money, sub], axis=1)
#parsed_date_money.to_excel('/root/projects/output/promiss/parsed_date_money_classification.xlsx')
parsed_date_money.to_json('/root/projects/output/promiss/parsed_date_money_classification.json')

# subs = ['submisssion10_3_17_309000.xlsx',
# 'submisssion10_29_5_555000.xlsx',
# 'submisssion10_56_3_131000.xlsx',
# 'submisssion11_22_43_973000.xlsx',
# 'submisssion11_50_11_564000.xlsx',
# 'submisssion12_18_5_76000.xlsx',
# 'submisssion12_48_11_562000.xlsx']
#
# def join_submisions (subs):
#
#     all_dfs = pd.DataFrame()
#     for i in subs:
#         temp = pd.read_excel('output_data/{}'.format(i))
#         all_dfs = pd.concat([all_dfs, temp], axis=0)
#
#     all_dfs.to_excel('pre_sub.xlsx', index=False)

### Скрипт для кластеризации

# def make_artm_clusterisation(col):
#
#     """Get artm theta matrixes"""
#
#     collection_test = pd.DataFrame(collection).iloc[test_index].reset_index()
#     arr_test=[]
#     for index_number, i in zip(collection_test['index'], collection_test[0]):
#         arr_test.append(str(index_number) +  ' |@default_class ' + unicode(i))
#     pd.DataFrame(arr_test,index=None).to_csv('leaver_vw_form_test.txt',sep='\t',encoding='UTF-8',index=False, header=None)
#     batch_vectorizer_test = artm.BatchVectorizer(data_path="leaver_vw_form_test.txt", data_format="vowpal_wabbit", target_folder="leaver_vw_form_test", batch_size=100)
#     T = 120  # количество тем
#     topic_names=["sbj"+str(i) for i in range(T)]
#     model_artm = artm.ARTM(num_topics=T,
#                            topic_names=topic_names,
#                            class_ids={'@default_class':700},
#                            num_document_passes=20,
#                            seed=79,
#                            reuse_theta=True,
#                            cache_theta=True,
#                            scores=[artm.TopTokensScore(name='top_tokens_score', num_tokens=30)],
#                            regularizers=[artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=-0.15)])
#     dictionary=artm.Dictionary(name='dictionary')
#     dictionary.gather(batch_vectorizer_test.data_path)
#     model_artm.initialize('dictionary')
#     dictionary.filter(min_tf=3)
#     model_artm.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
#     model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_def'))
#     model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_lab'))
#     model_artm.scores.add(artm.PerplexityScore(name='PerplexityScore',dictionary='dictionary'))
#     model_artm.scores.add(artm.TopTokensScore(name="top_words", num_tokens=15))
#     model_artm.fit_offline(batch_vectorizer=batch_vectorizer_test, num_collection_passes=40)
#     test_transformed = model_artm.transform(batch_vectorizer_test).T
#     test_transformed = test_transformed.reset_index().sort_values('index')
#     test_transformed = test_transformed.reset_index(drop=True)
#     del test_transformed['index']
#
#     print model_artm.score_tracker["PerplexityScore"].value
#
#     all_topics = []
#
#     for topic_name in model_artm.topic_names:
#         all_words = ''
# #         print topic_name +': '
#         tokens = model_artm.score_tracker["top_words"].last_tokens
#         for word in tokens[topic_name]:
#             all_words+=word
#             all_words+=' '
# #             print word
#         all_topics.append(all_words)
#
# #     !rmdirC:\Users\u_m0h08\!VOC\leaver_vw_form_test
#
#     return test_transformed, pd.DataFrame(all_topics)
#
# def make_submission_clusterisation (collection):
#
#     artm_df, topics = make_artm_clusterisation(collection)
#     artm_df.columns = topics.values
#     preds_all = artm_df
#     max_3_all =[]
#     for i in range(len(preds_all)):
#         nums = preds_all.iloc[i].values
#         max_3 = heapq.nlargest(3, nums)
#         max_3_all.append(max_3)
#     preds_all = pd.concat([preds_all.reset_index(drop=True), pd.DataFrame(max_3_all)], axis=1)
#     cols = preds_all.iloc[:,:preds_all.shape[1]-3].columns
#     target_cols=[0,1,2]
#     for i in cols:
#         for target_col in target_cols:
#             preds_all.loc[preds_all[i]==preds_all[target_col], 'max_value{}'.format(target_col)] = i
#     return preds_all[[0,1,2,'max_value0', 'max_value1', 'max_value2']]
#
# preds_all = make_submission_clusterisation (collection)
#
# uniq_index = pd.DataFrame(preds_all['max_value0'].unique()).reset_index()
#
# preds_all = pd.merge(preds_all, uniq_index, left_on='max_value0', right_on=0, how='left')
#
# input_test = pd.read_excel('input/08_zp.xlsx')
#
# preds_all = pd.concat([input_test, preds_all], axis=1)
#
# preds_all.to_excel('test_artm2.xlsx', index=False)
