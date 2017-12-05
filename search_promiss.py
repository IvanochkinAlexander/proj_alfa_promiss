#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from nltk import word_tokenize
import re
from pymorphy2 import MorphAnalyzer
from collections import deque
morph = MorphAnalyzer()

def text2int (textnum, numwords={}):

    """Replaces verbal worms to int"""
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

def basic_cleaning2(string):

    """Cleaning text from numbers and punctuation"""

    string = string.lower()
    string = re.sub('[\a-z\(\)\!\^\%\$\'\"\.;,-\?\{\}\[\]\\/]', ' ', string)
    string = re.sub(' +', ' ', string)
    return string

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
#             if re.search(u'[а-яА-Я]',words[i]):
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


def window(seq, n=2):
    it = iter(seq)
    win = deque((next(it, None) for _ in xrange(n)), maxlen=n)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win

def search_in_text (col):

    """Searches keywords in whole sentence"""

    all_values = ''
    for i in keywords_time:
        for counting in range(len(col)):
            for win in window(col, n=5):
                if i in win:
        #                 print (i)
                    all_values+=i
                    all_values+=' '
    #     return all_values

def search_to_exclude (col):

    """"""

    all_values = ''
    for i in to_exclude:
        if i in col:
            return u'исключаем по ключевым словам'

    # return all_values

def take_digits (col):

    """Returns digits into list"""

    col  = re.findall(r'\d+', col)
    all_dig = ''
    for i in col:
        all_dig += str(i)
    return all_dig

keywords_time = [

    u'минута',
    u'час',
    u'день',
    u'сутки'
]

to_exclude = [

u'необходимый',
u'разрешенный',
u'бесплатный',
u'бесплатно',
u'подразделение',
u'чтобы',
u'тарифный',
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
u'рекомендоваться',
u'можно',
u'действителен',
u'не_реже'

]

path = '/root/projects/output/promiss/parsed_date_money_classification.xlsx'

def post_process_data (path):

#     """Returns clients promisses"""

    df = pd.read_excel(path)
    df['small_text_trans'] = df['small_text'].apply(text2int)
    df['small_text_temp'] = df['small_text_trans'].apply(lambda x : x.split('_')[1].replace('[', '').replace(']','').replace('%', u'процент'))
    df['small_text_temp'] = df['small_text_temp'].apply(lambda text : re.sub(u'любой', '0', text))
    df['small_text_temp'] = df['small_text_temp'].apply(lambda text : re.sub(u'этот же', '0', text))
    df['small_text_temp'] = df['small_text_temp'].apply(lambda text : re.sub(u'тот же', '0', text))
    df['small_text_temp'] = df['small_text_temp'].apply(lambda text : re.sub(u'моментально', u'0 час', text))
    df['small_text_temp'] = df['small_text_temp'].apply(lambda text : re.sub(u'сразу', u'0 час', text))
    df['small_text_temp'] = df['small_text_temp'].apply(lambda text : re.sub(u'%', u' процент', text))
    df['small_text_temp'] = df['small_text_temp'].apply(lambda text : re.sub('%', u' процент', text))
    df['small_text_temp'] = df['small_text_temp'].apply(lambda text : re.sub(u'р.', u' рубль', text))


    collection = [wrk_words_wt_no(text) for text in df['small_text_temp']]
    df['small_text_lemm']  = pd.DataFrame(collection)
    df['small_text_lemm_list']  = df['small_text_lemm'].apply(review_to_wordlist)
    df['small_text_lemm_list'] = df['small_text_lemm_list'].apply(lambda m: [x for x in m if x != None])
    df['len'] = df['small_text_lemm_list'].apply(lambda x : len(x))
    df = df[df['len']>=5]
    df = df.reset_index(drop=True)

    all_index = []
    all_windows = []
    all_keywords = []
    all_text = []
    all_indexes_item = []

    for i in range(df.shape[0]):

        for w in window(df['small_text_lemm_list'][i], n=5):
            if len(list(w))==5:
                for val in w:
                    if val in keywords_time:
                        # all_indexes.append(list(w).index(val))
                    # if list(w).index(val)==2:
                        all_index.append(i)
                        all_keywords.append(val)
                        all_windows.append(' '.join(list(w)))
                        all_text.append(' '.join(df['small_text_lemm_list'][i]))
                        all_indexes_item.append(list(w).index(val))
                    # elif (2 not in all_indexes) & (3 in all_indexes):
                    #     all_index.append(i)
                    #     all_keywords.append(val)
                    #     all_windows.append(' '.join(list(w)))
                    #     all_text.append(' '.join(df['small_text_lemm_list'][i]))
                    # elif (2 not in all_indexes) & (4 in all_indexes):
                    #     all_index.append(i)
                    #     all_keywords.append(val)
                    #     all_windows.append(' '.join(list(w)))
                    #     all_text.append(' '.join(df['small_text_lemm_list'][i]))


            # else:
            #     for val in w:
            #         if val in keywords_time:
            #             # if list(w).index(val)==2:
            #             all_index.append(i)
            #             all_keywords.append(val)
            #             all_windows.append(' '.join(list(w)))
            #             all_text.append(' '.join(df['small_text_lemm_list'][i]))


    all_df = pd.DataFrame()
    list_of_lists = [all_index, all_keywords, all_windows, all_text, all_indexes_item]

    for i in list_of_lists:
        all_df = pd.concat([all_df, pd.DataFrame(i)],axis=1)

    all_df.columns = ('index', 'times', 'window_', 'text_', 'window_item_')
    df = df.reset_index()
    all_df_merged = pd.merge(all_df, df, on='index', how='left')
    df = all_df_merged
    df = df[~df['window_'].isnull()]
    df['digits'] = df['window_'].apply(lambda x : re.findall(r'\d+', x))
    df['true_digit'] = df['window_'].apply(take_digits)
    df['temp_splited'] = df['times'].apply(lambda x: x.split(' '))
    df['count_times'] = df['temp_splited'].apply(lambda x: len(x))
    df['count_digits'] = df['digits'].apply(lambda x: len(x))
    df['final_col'] = ''
    df.loc[((df['count_times']==1) & (df['count_digits']==1)), 'final_col'] = df['true_digit'].astype(unicode) + ' (' + df['times'].astype(unicode) + ')'
    df['criteria']= ''
    df.loc[~df['final_col'].isnull(), 'criteria'] = u'клиентское обещание'
    df.loc[df['final_col']=='', 'criteria'] = u'нет информации про время'
    df.loc[~df['money'].isnull(), 'criteria'] = u'продуктовая тема'
    df.loc[(df['final_col'].isnull()) & (df['money'].isnull()), 'criteria'] = u'нет информации про время'
    df.loc[~df['date'].isnull(), 'criteria'] = u'информация про даты'
    df.loc[(df['count_digits']>=2) & (df['count_times']>=3), 'criteria'] = u'несколько обещаний'
    df['temp_exclude'] = df['small_text_lemm_list'].apply(search_to_exclude)
    df.loc[~df['temp_exclude'].isnull(), 'criteria'] = u'условия продуктов'
    clients_promiss = df[df[u'final_col']!=''][['link',u'max_value0','small_text','final_col', 'window_','window_item_', 'criteria', 'temp_exclude', 'text_']]
    clients_promiss = clients_promiss.rename(columns={u'max_value0':u'Точка CJM','small_text':u'Текст', 'final_col':u'Обещание'})
    clients_promiss['temp'] = clients_promiss[u'Обещание'].apply(lambda x : int(x.split(' (')[0]))
    clients_promiss = clients_promiss[clients_promiss['temp']<=500]
    clients_promiss['value'] = clients_promiss[u'Обещание'].apply(lambda x : x.split(' ')[0])
    clients_promiss['value'] = clients_promiss['value'].astype(int)

    clients_promiss['metric'] = clients_promiss[u'Обещание'].apply(lambda x : x.split(' ')[1])
    clients_promiss['time'] = ''
    clients_promiss.loc[((clients_promiss['value']>30) & (clients_promiss['metric'] !=u'(минута)')), 'time'] = u'отсекаем по времени'
    clients_promiss.loc[((clients_promiss['value']>120) & (clients_promiss['metric'] ==u'(минута)')), 'time'] = u'отсекаем по времени'
    del clients_promiss['temp']

    clients_promiss['uid'] = clients_promiss['link'] + '_' + clients_promiss[u'Обещание'] + '_' + clients_promiss[u'Текст']
    clients_promiss['count'] = clients_promiss.groupby('uid')['link'].transform('count')
    clients_promiss['min'] = clients_promiss.groupby('uid')['window_item_'].transform('min')
    second_items = clients_promiss[clients_promiss['window_item_'] ==2]['uid']
    clients_promiss.loc[clients_promiss['uid'].isin(second_items), 'min'] = 2
    clients_promiss['check'] = (clients_promiss['window_item_'] == clients_promiss['min']).astype(int)
    clients_promiss = clients_promiss[clients_promiss['check']==1]
    clients_promiss['okey'] = ''
    clients_promiss.loc[(clients_promiss['metric']== u'(минута)') & (clients_promiss['criteria'] != u'клиентское обещание'), 'okey'] = 'okey'
    del clients_promiss['uid']

    print ('done check')
    clients_promiss['uid'] = clients_promiss[u'Обещание'] + '_' + clients_promiss[u'Текст']
    clients_promiss = clients_promiss.drop_duplicates('uid', keep='first')
    del clients_promiss['uid']
    clients_promiss.to_excel('../output/promiss/clients_promiss.xlsx', index=False)
    df.to_excel('../output/promiss/all_data_temp.xlsx', index=False)

post_process_data (path)
