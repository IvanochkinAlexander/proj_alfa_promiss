
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
    to_exclude = ['https://github.com/', 'http://www.nfa.ru/']
    urls_df = urls_df[~urls_df['domain'].isin(to_exclude)]
    urls_df = urls_df.drop_duplicates(subset=0, keep='first')
    concated = pd.read_excel('../output/promiss/concated.xlsx')
    urls_df = urls_df[~urls_df[0].isin(concated['url'].values)]
    urls_df.to_excel('../output/promiss/to_parse_selenium.xlsx', index=False)
    print (urls_df['domain'].unique())
    print (urls_df.shape[0])
    print (concated.shape[0])

to_parse_selenium()
