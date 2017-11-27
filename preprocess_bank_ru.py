#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd


def no_list (col):
    a= ''
    for i in col:
        a+=unicode(i)
        a+=' '
    return a

def process_all (path):
    temp = pd.read_json(path)
    temp = temp.fillna('0')

    for i in temp.columns:
        if i.startswith('field'):
            temp[i] = temp[i].apply(no_list)

    temp = temp.rename(columns={'field1':'check', 'field2':'rank','field3':'theme','field4':'review','field5':'date','field6':'comments','field7':'watched','field8':'bank'})

    int_cols = ['rank', 'comments','watched']

    for i in int_cols:
        temp[i] = temp[i].astype(int)

    temp['date'] = pd.to_datetime(temp['date'])

    print (temp['date'].min())
    print (temp['date'].max())

    cols_to_del = ['_template', '_type']

    for i in cols_to_del:
        del temp[i]

    # temp.to_csv('/root/portia_projects/output_data/bank_ru_processed.csv', encoding='utf-8')

    temp['all'] = temp['review'] + '_' + temp['theme']
    temp = temp.rename(columns={'all':u'Суть обращения'})

    temp.to_excel('/root/portia_projects/output_data/bank_ru_processed.xlsx')

    return temp

path= '/root/portia_projects/output_data/bank_ru_5000_all.json'

temp = process_all(path)
