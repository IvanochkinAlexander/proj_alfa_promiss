#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from nltk import word_tokenize
import re
from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()

def text2int (textnum, numwords={}):

    """"""
    if not numwords:

        units = [
        u'ноль', u'один', u'два', u'три', u'четыре', u'пять', u'шесть', u'семь', u'восемь',
        u'девять', u'десять', u'одиннадцать', u'двенадцать', u'триинадцать', u'четырнадцать', u'пятнадцать',
        u'шестнадцать', u'семнадцать', u'восемнадцать', u'девятнадцать'
    ]

        tens = ["", "", u"двадцать", u"тридцать", u"сорок", u"пятьдесят", u"шестьдесят", u"семьдесят", u"восемьдесят", u"девяносто"]

        scales = [u"сто", u"тысяча", u"миллион", u"миллиард", u"миллион"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'первый':1, 'второй':2, 'третий':3, 'пятый':5, 'восьмой':8, 'девятый':9, 'двенадцатый':12}
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    curstring = ""
    onnumber = False
    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    return curstring

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

    words=word_tokenize(sent.lower())
    try:
        arr=[]
        for i in range(len(words)):
            if re.search(u'[а-яА-Я]',words[i]):
#                 arr.append(stemmer.stem(words[i]))###стемминг
                arr.append(morph.parse(words[i])[0].normal_form)###лемматизация
        words1=[w for w in arr]
        words1=No_with_word(words1)
        return words1
    except TypeError:
        pass

def review_to_wordlist(review):

    """Convert collection to wordlist"""

    words = review.lower().split()
    words = [w for w in words]
    return(words)

def search_in_text (col):
    all_values = ''
    for i in keywords_time:
        if i in col:
            all_values+=i
            all_values+=' '
    return all_values

def search_to_exclude (col):
    all_values = ''
    for i in to_exclude:
        if i in col:
            return 1

    # return all_values

def take_digits (col):
    col  = re.findall(r'\d+', col)
    all_dig = ''
    for i in col:
        # try:
            # if i <=500:
            # try:
        all_dig += str(i)
        # except:
            # all_dig+=str(0)
        # except:
            # all_dig += 0
    return all_dig

keywords_time = [
    u'секунда',
    u'минута',
    u'час',
    u'день',
    u'сутки',
    u'неделя',
    u'месяц'
]

to_exclude = [

u'необходимый',
u'разрешенный',
u'бесплатный',
u'бесплатно',
u'подразделение',
u'чтобы',
u'процент',
u'действовать',
u'можно',
u'подойти',
u'следовать',
u'не_менее',
u'подробный',
u'подписка',
u'комиссия',
u'скидка',
u'погашение',
u'погашеный',
u'погашен',
u'снятие',
u'расторжение',
u'размещение',
u'пополнение',
u'подарок',
u'должный',
u'должен',
u'скидка'

]

def post_process_data (path):

    df = pd.read_excel(path)
    df['small_text_trans'] = df['small_text'].apply(text2int)
    df['small_text_temp'] = df['small_text_trans'].apply(lambda x : x.split('_')[1].replace('[', '').replace(']','').replace('%', u'процент'))
    collection = [wrk_words_wt_no(text) for text in df['small_text_temp']]
    df['small_text_lemm']  = pd.DataFrame(collection)
    df['small_text_lemm_list']  = df['small_text_lemm'].apply(review_to_wordlist)
    df['times'] = df['small_text_lemm_list'].apply(search_in_text)
    df['digits'] = df['small_text_trans'].apply(lambda x : re.findall(r'\d+', x))
    df['true_digit'] = df['small_text_trans'].apply(take_digits)
    df['temp_splited'] = df['times'].apply(lambda x: x.split(' '))
    df['count_times'] = df['temp_splited'].apply(lambda x: len(x))
    df['count_digits'] = df['digits'].apply(lambda x: len(x))
    # df.loc[((df['true_digit'].astype(int)>=500) & (df['count_digits']==2)), 'true_digit'] = np.nan
    df['final_col'] = ''
    df.loc[((df['count_times']==2) & (df['count_digits']==1)), 'final_col'] = df['true_digit'].astype(unicode) + ' (' + df['times'].astype(unicode) + ')'
    df['criteria']= ''
    df.loc[~df['final_col'].isnull(), 'criteria'] = u'клиентское обещание'
    df.loc[df['final_col']=='', 'criteria'] = u'нет информации про время'
    df.loc[~df['money'].isnull(), 'criteria'] = u'продуктовая тема'
    df.loc[(df['final_col'].isnull()) & (df['money'].isnull()), 'criteria'] = u'нет информации про время'
    df.loc[~df['date'].isnull(), 'criteria'] = u'информация про даты'
    df.loc[(df['count_digits']>=2) & (df['count_times']>=3), 'criteria'] = u'несколько обещаний'
    df['temp_exclude'] = df['small_text_lemm_list'].apply(search_to_exclude)
    df.loc[~df['temp_exclude'].isnull(), 'criteria'] = u'условия продуктов'

    clients_promiss = df[df['criteria']==u'клиентское обещание'][['link',u'max_value0','small_text',u'previous and next','final_col']]
    clients_promiss = clients_promiss.rename(columns={u'max_value0':u'Точка CJM','small_text':u'Текст',  u'previous and next':u'Контекст','final_col':u'Обещание'})
    clients_promiss['temp'] = clients_promiss[u'Обещание'].apply(lambda x : int(x.split(' (')[0]))
    clients_promiss = clients_promiss[clients_promiss['temp']<=500]
    del clients_promiss['temp']
    clients_promiss.to_excel('../output/promiss/clients_promiss.xlsx', index=False)
    df.to_excel('../output/promiss/all_data_temp.xlsx', index=False)

path = '/root/projects/output/promiss/parsed_date_money_classification.xlsx'
post_process_data (path)
