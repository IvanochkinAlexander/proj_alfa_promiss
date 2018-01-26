import pandas as pd
from natasha.grammars import Person, Organisation, Address, Street,Money, Date
from natasha import Combinator

def get_url (col):
    try:
        col = col.split('.ru')[0].split('https://')[1]
        if col.startswith('www.'):
            col = col.split('www.')[1]
        return col
    except:
        try:
            col = col.split('.ru')[0].split('http://')[1]
            if col.startswith('www.'):
                col = col.split('www.')[1]
            return col
        except:
            return 'empty'


def unique_list(l):

    """Takes unique words from parsed string"""

    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


def parse_one_page (link, name_of_link):

    """Parses keywords from the text"""

    parsed_one = pd.DataFrame()
    text=link

    combinator = Combinator([
        Money,
        Date
    ])

    matches = combinator.resolve_matches(
        combinator.extract(text), strict=False
    )
    matches = (
        (grammar, [t.value for t in tokens]) for (grammar, tokens) in matches
    )

    text_money = ''
    text_date = ''

    for i in matches:

        if str(i[0]).startswith('Date'):
            temp_value = ''
            for value in i[1]:
                temp_value+=str(value)
                temp_value+=' '
            text_date +=temp_value

        elif str(i[0]).startswith('Money'):
            temp_value = ''
            for value in i[1]:
                temp_value+=str(value)
                temp_value+=' '
            text_money +=str(temp_value)

    text_money=' '.join(unique_list(text_money.split()))
    text_date=' '.join(unique_list(text_date.split()))
    parsed_one = pd.DataFrame()
    parsed_one = pd.DataFrame([name_of_link, link, text_date, text_money]).T
    parsed_one.columns = ('name', 'link',  'date', 'money')
    return parsed_one

def parse_all (train_test):
    all_temp = pd.DataFrame()
    for i, k in zip(train_test['small_text'].values, train_test['link'].values):
        temp = parse_one_page(i, k)
        all_temp=pd.concat([all_temp, temp],axis=0)
    del all_temp['link']
#     all_temp.to_excel('../output/temp_parse.xlsx', index=False)
    print ('Done.')
    return all_temp

df = pd.read_json('../output/promiss/all_text_from_multiple.json')
print ('loaded')
# df = pd.read_excel('../output/promiss/all_text_from_multiple.xlsx')
df['bank'] = df['link'].apply(get_url)
train_test = df
all_temp = parse_all (train_test)
temp = all_temp[['date', 'money']].reset_index(drop=True)
df = pd.concat([df.reset_index(drop=True), temp], axis=1)
df.to_json('../output/promiss/parsed_date_money.json')
df = df.rename(columns={'small_text':u'Суть обращения'})
df['month']=11
df[['Суть обращения', 'month']].to_json('../output/promiss/test.json')
# df[['Суть обращения']].to_json('../output/promiss/test.json')
print ('сохранили файлы')
