from pymorphy2 import MorphAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re
import pandas as pd
import numpy as np
import time
import time
import re
import datetime
from nltk.corpus import stopwords
from nltk.util import ngrams
import time
start_time = time.time()

def No_with_word(token_text):

    """Concat no part"""

    tmp=[]
    for i,word in enumerate(token_text):
        if word==u'не':
            tmp.append("_".join(token_text[i:i+2]))
        else:
            if token_text[i-1]!=u'не':
                tmp.append(word)
    return tmp

def wrk_words_wt_no(sent):

    """Make ngrams with lemmatisation"""

    words=word_tokenize(sent.lower())
    lemmas = []
    values = []
    values_all = []
    try:
        arr=[]
        for i in range(len(words)):
            if re.search(u'[а-яА-Я]',words[i]):
                arr.append(morph.parse(words[i])[0].normal_form)
            # lemmas=[w for w in arr if w not in stop]


            # print (lemmas)
            #     values = [" ".join(value) for value in list(ngrams(lemmas,2))]
            # values_all.append(values)

        # return words
            # lemmas=[w for w in arr if w not in stop]
            # values = [" ".join(value) for value in list(ngrams(lemmas,2))]

        # values_all.append(words)
            # values_all.append(values)
        # return values

        lemmas=[w for w in arr]
        words = No_with_word(lemmas)
        # return words
        # lemmas=[w for w in arr]
        values = [" ".join(value) for value in list(ngrams(lemmas,2))]
        return values, words
    except TypeError:
         pass

def split_sentences (text_df):

    """Split texts into sentences"""

    text_df = text_df.rename(columns={'text':'full_text'})
    text_df = text_df[~text_df['full_text'].isnull()]
    text_df['type'] = text_df['full_text'].apply(lambda x:type(x))
    morph = MorphAnalyzer()

    keysentenses_all=[]

    for i in range(len(text_df)):
        try:
            sentences=sent_tokenize(text_df['full_text'][i])
        except:
            pass
        for sent in sentences:
            keysentenses_all.append(sent)

    return keysentenses_all, text_df


# def find_keywords (text_df, keysentenses_all, to_search):
#
#     """Searching keywords in sentences"""
#
#     keysentenses=[]
#     keysentenses2=[]
#
#     for i in range(len(text_df)):
#         keysentenses.append(i)
#         keysentenses2.append(i)
#         temp_list = []
#         try:
#             sentences=sent_tokenize(text_df['full_text'][i])
#             for k, sent in enumerate(sentences):
#                 words, values = wrk_words_wt_no(sent)
#                 total_list = words+values
#                 for token in total_list:
#                     # print (token)
#                     # temp_list.append(token)
#                     for keyword in to_search:
#                         if token == keyword:
#                             if k ==0:
#                                 try:
#                                     keysentenses.append(keyword+'_' + sent)
#                                     keysentenses2.append(keyword+'_' + sent + keysentenses_all[k+1])
#                                 except:
#                                     keysentenses2.append(keyword+'_' + sent)
#                             else:
#                                 try:
#                                     keysentenses.append(keyword+'_' + sent)
#                                     keysentenses2.append(keyword+'_' + keysentenses_all[k-1] + sent + keysentenses_all[k+1])
#                                 except:
#                                     keysentenses2.append(keyword+'_' + keysentenses_all[k-1] + sent)
#                         else:
#                             pass
#         except:
#             pass
#     parsed_data = pd.DataFrame(keysentenses)
#     parsed_data2 = pd.DataFrame(keysentenses2)
#     print (parsed_data.shape[0])
#     print (parsed_data2.shape[0])
#
#
#
#
#     if parsed_data.shape[0] ==parsed_data2.shape[0]:
#         parsed_data = pd.concat([parsed_data, parsed_data2], axis=1)
#     parsed_data.columns = (0, 'previous and next')
#     parsed_data = parsed_data.drop_duplicates(subset=0, keep='first')
#     parsed_data['shifted'] = parsed_data[0].shift(1).fillna(0)
#     parsed_data.loc[parsed_data[0].isin(range(len(text_df))), 'shifted'] = parsed_data[0]
#     parsed_data['type'] = parsed_data['shifted'].apply(lambda x:type(x))
#
#     return parsed_data


def find_keywords (text_df, keysentenses_all, to_search):

    """Searching keywords in sentences"""

    keysentenses=[]
    keysentenses2=[]

    for i in range(len(text_df)):
        keysentenses.append(i)
        keysentenses2.append(i)
        temp_list = []
        # try:
        sentences=sent_tokenize(text_df['full_text'][i])
        for k, sent in enumerate(sentences):
            words, values = wrk_words_wt_no(sent)
            total_list = words+values
            # for token in total_list:
            #     # print (token)
            #     # temp_list.append(token)
            #     for keyword in to_search:
            #         if token == keyword:
            #             if k ==0:
            try:
                keysentenses.append('keyword'+'_' + sent)
                keysentenses2.append('keyword'+'_' + sent + keysentenses_all[k+1])
            except:
                keysentenses2.append('keyword'+'_' + sent)

    parsed_data = pd.DataFrame(keysentenses)
    parsed_data2 = pd.DataFrame(keysentenses2)
    print (parsed_data.shape[0])
    print (parsed_data2.shape[0])

    if parsed_data.shape[0] ==parsed_data2.shape[0]:
        parsed_data = pd.concat([parsed_data, parsed_data2], axis=1)
    parsed_data.columns = (0, 'previous and next')
    parsed_data = parsed_data.drop_duplicates(subset=0, keep='first')
    parsed_data['shifted'] = parsed_data[0].shift(1).fillna(0)
    parsed_data.loc[parsed_data[0].isin(range(len(text_df))), 'shifted'] = parsed_data[0]
    parsed_data['type'] = parsed_data['shifted'].apply(lambda x:type(x))

    return parsed_data

def post_processing (parsed_data, text_df):

    """Preprocess data"""

    for i in range(100):
        parsed_data.loc[parsed_data['type'] != int,  'shifted'] = parsed_data['shifted'].shift(1)
    parsed_data['type'] = parsed_data[0].apply(lambda x:type(x))
    # (parsed_data['type']==unicode) |
    parsed_data = parsed_data[(parsed_data['type']==str)]
    text_df = text_df.reset_index()
    text_df = text_df.rename(columns={'index':'level_0', 'url':'link'})
    merged_data = pd.merge(parsed_data, text_df, left_on='shifted', right_on='level_0', how='left')
    merged_data_cut = merged_data[[u'link', u'full_text', 0, 'previous and next']]
    merged_data_cut = merged_data_cut.rename(columns={0:'small_text'})
    merged_data_cut['small_text'] = merged_data_cut['small_text'].apply(lambda x :x.strip().replace('\n', '').replace('\t', ''))
    merged_data_cut['previous and next'] = merged_data_cut['previous and next'].apply(lambda x :x.strip().replace('\n', '').replace('\t', ''))
    merged_data_cut['temp'] = merged_data_cut['small_text'].apply(lambda x:x.split('_')[1])
    print ('с дубликатами составляет_{}'.format(merged_data_cut.shape[0]))
    merged_data_cut = merged_data_cut.drop_duplicates(subset='temp', keep='first')
    merged_data_cut['keyword'] = merged_data_cut['small_text'].apply(lambda x:x.split('_')[0])
    print (merged_data_cut.shape[0])
    date_time = datetime.datetime.now()
    merged_data_cut.to_excel('../output/promiss/all_text_from_multiple.xlsx', index=False)
    return merged_data_cut


# def run_keywords ():

to_search = [
u'дата',u'день',u'срок',u'заявка',u'платёж',u'быстро',u'время',u'месяц',
u'условие'u'гибко',u'самое',u'своевременно обработка',u'точно',u'информирование',u'качественно',u'удобно',
u'удаленно',u'понятно',u'хорошо',u'отлично',
u'в течение',u'срок открытие',u'срок расмотрение',
u'срок ответа',u'время ожидание',u'срок перевода',u'срок предоставление',u'время обслуживание',u'срок погашение',
u'время поступление',u'срок поступление',u'срок выдача',u'срок закрытие',u'срок выпуск',u'срок перевыпуск',
u'срок подключение',u'время обращение',u'срок обработка',u'срок перечисление',u'срок снятие',u'срок зачисление',
u'время оформление',u'рабочий день',u'в срок',u'круглосуточно',u'первый раз',u'операционный день',u'любое отделение',
u'любой сотрудник',u'без очереди',u'любой специалист',u'без проблема',u'без ожидание',u'любой время',u'один документ',
u'два документа',u'время реагирование',u'без привязка',u'без потеря',u'рабочий день',u'срок действие',u'срок ведение',
u'дата подача',u'окончание срок',u'календарный день',u'срок обслуживание',u'дата обращение',u'следующий рабочий',
u'дата окончание',u'дата поступление',u'обработка заявка',u'дата подключение',u'снятие наличные',u'дата открытие',
u'перевыпуск карта',u'срок предоставление',u'течение срок',u'срок регистрация',u'дата закрытие',u'дата предоставление',
u'доставка документ',u'дата выдача',u'поступление заявка',u'следующий день',u'момент получение',u'отчётный период',
u'текущий день',u'открытие счёт',u'закрытие счёт',u'подача заявка',u'прекращение действие',u'город присутствие',
u'выдача наличные',u'регион присутствие',u'ближний рабочий',u'получить возможность',u'есть возможность' u'проведение операция',u'поступление звонок']

# mystopwords = stopwords.words('russian')+[u'это', u'иза', u'свой',u'млрд', u'млн',u'млна',u'тыс',\
#                                           u'трлн', u'вопрос', u'весь', u'который', u'наш', '-', ',',\
#                                           u'это', u'вопрос', u'весь', u'самый', u'ваш', u'наш', u'почему', u'свой',\
#                                          '=', '{', '+', '}', 'var', u'–', '1', 'if', '/', '5', u'г.', '});', '0;', 'return', 'i', '>', 'listid', 'isfavorite',\
#                                          'false;', 'webid', 'result', 'function(data)', '2', '3', '4', '5', '6', '7', '8', '9', 'url', u'(в',\
#                                          'function', '000', 'window.tops.constants.contenttypes.akbusermanual', U'(переть', u'случае,', '10', '12',\
#                                          u'ucs.ести', u'«мыть', u'(ить', u'(переть', u'(ести', u'существующих)*ести', u'(возникать', u'(мочь', \
#                                          u'который', u'данный']
### executing script
# stop = mystopwords
stop= []
# stop.pop(stop.index(u'не'))
morph = MorphAnalyzer()
text_df = pd.read_excel('../output/promiss/concated.xlsx')
keysentenses_all, text_df = split_sentences(text_df)
parsed_data= find_keywords (text_df, keysentenses_all, to_search)
merged_data_cut = post_processing(parsed_data, text_df)

print("--- %s seconds ---" % (time.time() - start_time))

# if __name__ == '__main__':
#     # find_keywords.py executed as script
#     # do somethings
#     run_keywords()
