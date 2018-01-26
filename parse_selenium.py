#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymorphy2 import MorphAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup as bs
import urllib
import re
import pandas as pd
import numpy as np
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from collections import  Counter
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import re
import datetime
# from cStringIO import StringIO
# from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
# from pdfminer.converter import TextConverter
# from pdfminer.layout import LAParams
# from pdfminer.pdfpage import PDFPage
import subprocess
import telegram
import time

def send_to_telegram(text):

    """Send appropriate links to telegram channel"""

    bot = telegram.Bot(token='379005601:AAH1rv3ESXLWTXbn14gnCxW52eeKc4qnw50')
    # chat_id = -1001111732295
    chat_id = 169719023
    bot.send_message(chat_id=chat_id, text=text)
    time.sleep(10)

date_time = datetime.datetime.now()

random_tags = ["panel",
              "news-text news-text-html",
              "layout__content content","sbrf-rich-wrapper",
              "content",
              "personalizable-container__cell col-xs-12 col-sm-6 col-md-3 personalizable-container__cell_occupied",
              "bp-widget-body",
              "card-info",
              "content-text",
              "mid_col",
              "wrapper",
              "wrapper__content",
              "section",
             "ui-trading-stock-about",
             "news-about__insurance",
             "news-about__desc  static-text",
             "bull",
             "columns__col  columns__col_content",
             "panel panel_news-about",
             "news-list__item  news-list__item_right-list  ",
             "columns__col panel columns__col_sidebar",
             "ui-collapse-banner__inner ui-collapse-banner__inner_children TCS-menu-float",
             "ui-action-menu",
             "ui-menu-second ui-menu-second_product",
             "ui-menu-second__layout",
             "ui-menu-second__wrapper ui-menu-second__wrapper_product",
             "ui-menu-second__container ui-layout__container ui-menu-second__container_product",
             "ui-scroll ui-menu-second__list ui-menu-second__list_product",
             "ui-scroll__window",
             "ui-header-primary__column ui-header-primary__column_toggle",
             "ui-header-primary__column ui-header-primary__column_menu",
             "ui-trading-news-list__content-announce",
             "ui-trading-security-news",
             "ui-trading-security-news__filte",
             "ui-context-filter ui-context-filter_filter-news",
             "application",
	     "tb-container",
             "div",
             "col-group",
             "main__inner main__inner_no-bg",
             "services-payment__item-text",
             "main_text",
             "b-container",
             "body-container",
             "descrip"
             ]

def cut_data (links_df):

    """Reduces number of samples by 150 and cuts long domain names"""

    links_df['length'] = links_df[0].apply(lambda x: x.count('/'))
    links_df['count'] = links_df.groupby('domain')[0].transform('count')
    links_df = links_df.sort_values(['domain', 'length'])
    final_df = pd.DataFrame()
    for i in links_df['domain'].unique():
        links_df_df = links_df[links_df['domain']==i]
        links_df_df= links_df_df[:300]
        final_df= pd.concat([final_df, links_df_df])
    final_df = final_df.reset_index(drop=True)
    final_df = final_df[final_df['length']<=9]
    return final_df

# def parse_links (links_alfa, random_tags):
#
#     """Parse all links"""
#
#     long_text = []
#     values = []
#     for i, k in enumerate(pd.DataFrame(links_alfa)[0].unique()):
#         try:
#             print (k)
#
#             YOUR_PAGE_URL = k
#             browser.get(YOUR_PAGE_URL)
#             page = browser.page_source
#             soup = BeautifulSoup(page, "lxml")
#             long_text.append(i)
#             long_text.append(soup.text)
#         except:
#             print ('ошибка')
#             pass
#     return long_text


def parse_links (links_alfa, random_tags):

    """Parse all links"""

    long_text = []
    values = []
    for i, k in enumerate(pd.DataFrame(links_alfa)[0].unique()):
        try:
            print (k)
            YOUR_PAGE_URL = k
            browser.get(YOUR_PAGE_URL)
            page = browser.page_source
            soup = BeautifulSoup(page, "lxml")
            for value in soup.find_all('div', class_=random_tags):
                long_text.append(i)
                long_text.append(value.text)
            # long_text.append(i)
            # long_text.append(soup.text)
        except:
            print ('ошибка')
            pass
    return long_text

def post_processing (long_text):

    """Convert parsed links to dataframe"""

    text_down = pd.DataFrame(long_text)
    text_down['shifted'] = text_down[0].shift(1)
    text_down['type'] = text_down[0].apply(lambda x:type(x))
    text_down = text_down[(text_down['type']==str)]
    text_down=text_down.rename(columns={0:'text'})
    links_df = pd.DataFrame(pd.DataFrame(links_alfa)[0].unique()).reset_index()
    merged_text = pd.merge(text_down, links_df, left_on='shifted', right_on='index', how='left')
    merged_text_cut = merged_text[['text', 0]]
    merged_text_cut.columns=('full_text', 'link')
    merged_text_cut = merged_text_cut.drop_duplicates(subset='full_text', keep='first')
    merged_text_cut['full_text'] = merged_text_cut['full_text'].astype(str)
    merged_text_cut.columns = ('text', 'url')
    merged_text_cut[['url', 'text']].to_json('../output/promiss/selenium_parsed.json')
    send_to_telegram('пропарсено ссылок selenium!!! {}'.format(merged_text_cut['url'].nunique()))
    print (merged_text_cut.shape[0])


links_alfa = pd.read_json('../output/promiss/to_parse_selenium.json')
browser = webdriver.PhantomJS(executable_path='../../../usr/local/bin/phantomjs')
browser.set_page_load_timeout(40)
send_to_telegram(('всего ссылок без удаления {}'.format(links_alfa.shape[0])))
print (links_alfa.columns)
links_alfa=links_alfa.rename(columns={'0':0})
links_alfa = cut_data(links_alfa)
links_alfa.to_json('../output/promiss/to_parse_selenium.json')
send_to_telegram(('после удаления по условиям {}'.format(links_alfa.shape[0])))
long_text = parse_links(links_alfa, random_tags)
post_processing (long_text)
send_to_telegram ('если это сообщение пришло, значит сработал селениум')
