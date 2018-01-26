#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import telegram
import time


def send_to_telegram(text):

    """Send appropriate links to telegram channel"""

    bot = telegram.Bot(token='379005601:AAH1rv3ESXLWTXbn14gnCxW52eeKc4qnw50')
    # chat_id = -1001111732295
    chat_id = 169719023
    bot.send_message(chat_id=chat_id, text=text)
    time.sleep(10)

def read_transform (path):

    """Reading and tfidf vactorisation"""

    temp = pd.read_json(path)
    send_to_telegram(u'Длина полного датафрейма составляет: {}'.format(temp.shape[0]))
    time.sleep(3)
    temp['ko'] = ''
    # temp.loc[temp['okey']!='okey', 'ko'] = 1
    temp['okey'].fillna('empty', inplace=True)
    temp.loc[temp['okey']=='okey', 'ko'] = 1
    temp.loc[temp['criteria']==u'клиентское обещание', 'ko'] = 1
    temp.loc[temp['time']==u'отсекаем по времени', 'ko'] = ''
    temp = temp[temp['ko']==1]
    send_to_telegram(u'Длина отфильтрованного датафрейма составлет: {}'.format(temp.shape[0]))
    time.sleep(3)
    temp['uid'] = temp['link'] + '_' + temp[u'Обещание']
    temp['count_uid'] = temp.groupby('uid')['link'].transform('count')
    temp_cut = temp[temp['count_uid']>=2]
    tf = TfidfVectorizer()
    tfidf = tf.fit_transform(temp_cut['text_'])
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=tf.get_feature_names())
    return tfidf_df, temp_cut, temp

def compare_tfidfs (tfidf_df):

    """Count cos similiarity acoording to tf-idf"""

    sims_df = pd.DataFrame(cosine_similarity(tfidf_df))
    for i in range(len(sims_df)):
        sims_df.iloc[i,i] = 0
    final_list = []
    for i in sims_df.columns:
        one_col = pd.DataFrame(sims_df.iloc[:,i])
        one_col = one_col[one_col[i]>=0.6]
        temp_list = list(one_col.index)
        temp_list.append(i)
        temp_list = sorted(temp_list)
        temp_str = ' '.join(str(k) for k in temp_list)
        final_list.append(temp_str)
    final_df = pd.DataFrame(final_list)
    return final_df

def concat_and_save (final_df, temp, temp_cut):

    """Deleting duplicates"""

    final_df = final_df.set_index(temp_cut.index)
    temp = pd.concat([temp, final_df], axis=1)
    temp['uid_2'] = temp['uid'] + '_' + temp[0]
    temp.loc[temp['uid_2'].isnull(), 'uid_2'] = temp['uid']
    # send_to_telegram(u'Строчек до удаления дубликатов: {}'.format(temp.shape[0]))
    time.sleep(3)
    to_del = ['time', 'count', 'min', 'check', 'okey', 'ko', 'uid',  0, 'text_', 'temp_exclude', 'criteria', 'window_', 'window_item_']
    for i in to_del:
        del temp[i]
    temp.to_json('../output/promiss/with_duplicates.json')
    temp = temp.drop_duplicates(subset='uid_2', keep='first')
    temp.to_json('../output/promiss/no_duplicates.json')
    send_to_telegram(u'Строчек после удаления дубликатов: {}'.format(temp.shape[0]))
    time.sleep(3)

### executing script
path = '../output/promiss/clients_promiss.xlsx'
tfidf_df, temp_cut, temp = read_transform (path)
final_df = compare_tfidfs (tfidf_df)
concat_and_save (final_df, temp, temp_cut)
