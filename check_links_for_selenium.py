
#!/usr/bin/env python3

import subprocess
import os
import telegram
import time
import pandas as pd
import re
from urllib.parse import urlparse


def read_parse_url (path):

    """Exctacting domain links from the text"""

    file = open(path, 'r')
    url = file.read()
    file.close()

    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', url)
    urls_df = pd.DataFrame(urls)
    urls_df[0] = urls_df[0].apply(lambda x : x.replace('>', '').replace(')', '').replace("'", "").replace(",", ""))
    # urls_df.to_excel('../output/promiss/temp_to_del.xlsx', index=False)

    return urls_df

def parse_domain_name (col):

    """Exctracting domain name from url ling"""

    parsed_uri = urlparse(col)
    domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)

    return domain

def to_parse_selenium ():

    """
    Some urls are not parsed by Scrapy,
    we are looking for this links and
    prepare to parse them by Selenium with PhantomJs
    """

    path = 'temp'
    urls_df = read_parse_url(path)
    print (urls_df.shape[0])
    urls_df['domain'] = urls_df[0].apply(parse_domain_name)
    print (urls_df['domain'].unique())
    urls_df['http'] = urls_df[0].apply(lambda x : x.split(':/')[0])
    urls_df.to_excel('../output/promiss/temp_to_del.xlsx', index=False)
    urls_df.loc[urls_df['domain']=='http://www.gazprombank.ru/', 'http'] = 'https'
    urls_df=urls_df[urls_df['http']=='https']

    to_exclude = ['https://github.com/',
    'http://www.nfa.ru/',
    'https://anketa.alfabank.ru/',
    'https://anketa.bm.ru/',
    'https://bg.modulbank.ru/',
    'https://bmmobile.bm.ru/',
    'https://cabinet.mdmbank.ru/',
    'https://click.alfabank.ru/',
    'https://click.alfabank.ru:443/',
    'https://correqts.kedrbank.com/',
    'https://dom.gosuslugi.ru/',
    'https://dom.gosuslugi.ru:/',
    'https://falcon.binbank.ru/',
    'https://hh.ru/',
    'https://hr.alfabank.ru/',
    'https://ibank.alfabank.ru/',
    'https://kpu.alfabank.ru/',
    'https://link.alfabank.ru/',
    'https://lk2.service.nalog.ru/',
    'https://online.alfabank.ru/',
    'https://online.sberbank.ru/',
    'https://pay.alfabank.ru/',
    'https://pos.modulbank.ru/',
    'https://potok.digital/',
    'https://reg-alfabank.ru/',
    'https://reg.alfabank.ru/',
    'https://reg.alfabank.ru:443/',
    'https://rosreestr.ru/',
    'https://ru.unicreditbanking.net/',
    'https://ru.wargaming.net/',
    'https://rzd-bonus.ru/',
    'https://sberbankins.ru/',
    'https://sbi.sberbank.ru:9443/',
    'https://sense.alfabank.ru/',
    'https://spasibosberbank.travel./',
    'https://spasibosberbank.travel/',
    'https://support.unicredit.ru/',
    'https://webquik.sberbank.ru/',
    'https://www.aviacompensation.ru/',
    'https://www.sberbank-insurance.ru/',
    'https://www.sberbank.ru:443/',
    'https://zp.alfabank.ru/']

    urls_df = urls_df[~urls_df['domain'].isin(to_exclude)]
    urls_df = urls_df.drop_duplicates(subset=0, keep='first')
    concated = pd.read_excel('../output/promiss/concated.xlsx')
    urls_df = urls_df[~urls_df[0].isin(concated['url'].values)]
    urls_df.to_excel('../output/promiss/to_parse_selenium.xlsx', index=False)

    print (urls_df['domain'].unique())
    print (urls_df.shape[0])
    print (concated.shape[0])

to_parse_selenium()
