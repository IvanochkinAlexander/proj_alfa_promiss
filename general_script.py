import subprocess
import os
import telegram
import time
import pandas as pd


def send_to_telegram(text):

    """Send appropriate links to telegram channel"""

    bot = telegram.Bot(token='')
    chat_id = ''
    bot.send_message(chat_id=chat_id, text=text)
    time.sleep(15)


def read_and_concat (list_of_bank):

    """Read all target files and concat"""

    all_df = pd.DataFrame()
    for file_name in list_of_bank:
        try:
            print (file_name)
            temp = pd.read_json('~/portia_projects/output_data/{}.json'.format(file_name))
            print (temp.shape[0])
            temp.fillna(0, inplace=True)

            field_col = []
            for i in temp.columns:
                if i.startswith('field'):
                    field_col.append(i)

            all_text = []
            for i in range(temp.shape[0]):
                temp_text = ''
                for k in field_col:
                    str_value=str(temp.loc[i, k])
                    if str_value != '0':
                        temp_text+=str_value
                        temp_text+='. '
                all_text.append(temp_text)
        except:
            pass

        final_df = pd.concat([temp[['url']], pd.DataFrame(all_text)],axis=1)
        final_df = final_df.rename(columns={0:'text'})
        all_df = pd.concat([all_df, final_df])
        all_df = all_df.reset_index(drop=True)
    return all_df

def run_parsing (list_of_bank):

    """run total script"""


    for bank in list_of_bank:

        send_to_telegram('парсится файл {}'.format(bank))

        try:
            path = '/root/portia_projects/output_data/{}.json'.format(bank)
            os.remove(path)

            print ('ok')
        except OSError:
            pass

    subprocess.check_output('docker run -i --rm -v ~/portia_projects:/app/data/projects:rw -v ~/portia_projects/output_data:/mnt:rw -p 9003:9003 scrapinghub/portia \
    portiacrawl -s USER_AGENT="Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36" -s depth_limit=3 -s download_timeout=15 /app/data/projects/new_rb {}'.format(bank) + ' -o /mnt/{}.json'.format(bank), shell=True)
    subprocess.check_output('python ../../python-sitemap/main.py --domain https://www.tinkoff.ru/ --output ../output/promiss/sitemap.xml --exclude "invest" --exclude "news" --exclude "login" --exclude "eng" --exclude "auth" --exclude "about" --verbose', shell=True)
    send_to_telegram('сделали tcs')
    subprocess.check_output('python ../../python-sitemap/main.py --domain https://open.ru/ --output sitemap.xml --exclude "storage" --exclude "about" --exclude "news" --exclude "press" --exclude "addresses" --verbose', shell=True)
    send_to_telegram('сделали open')
    subprocess.check_output('python ../../python-sitemap/main.py --domain https://rosbank.ru/ --output sitemap.xml --exclude "files" --exclude "offices" --exclude "pm" --exclude "upload" --exclude "about" --exclude "en" --exclude "realty" --exclude "atm" --exclude "search" --exclude "fin_inst" --exclude "promo" --exclude "press" --verbose', shell=True)
    send_to_telegram('сделали rosbank')
    n=60
    concated = read_and_concat(list_of_bank)
    concated=concated.reset_index(drop=True)
    concated.to_json('../output/promiss/concated.json')
    send_to_telegram('соединили файлы')
    time.sleep(n-50)

    process = subprocess.Popen("python /root/projects/proj_alfa_promiss/check_links_for_selenium.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    errcode = process.returncode

    send_to_telegram('проверили ссылки')
    time.sleep(n)

    process = subprocess.Popen("python /root/projects/proj_alfa_promiss/parse_selenium.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    errcode = process.returncode
    
    send_to_telegram ('пропарсили selenium')

    time.sleep(n+10)
    parsed_selenium = pd.read_json('../output/promiss/selenium_parsed.json')
    time.sleep(10)
    send_to_telegram ('длина парсед селениум_только селениум_{}'.format(parsed_selenium.shape[0]))
    concated = pd.concat([concated, parsed_selenium])
    concated = concated[~concated['text'].isnull()]
    concated = concated[concated['text'] != '']
    concated=concated.reset_index(drop=True)
    send_to_telegram ('длина конкатед_crapy+selenium_{}'.format(concated.shape[0]))
    concated.to_json('../output/promiss/concated.json')

    time.sleep(n-20)

list_of_bank = ['pcht', 'www.rshb.ru', 'alfabank.ru', 'mb','uc','vtb','mcb','bnb','open', 'rf', 'ps', 'sb', 'gb']

n=60

run_parsing(list_of_bank)

process = subprocess.Popen("python /root/projects/proj_alfa_promiss/find_keywords.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()
errcode = process.returncode


send_to_telegram('пропарсили ключевые слова')
time.sleep(n+30)

process = subprocess.Popen("python /root/projects/proj_alfa_promiss/parse_date_money.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()
errcode = process.returncode


send_to_telegram ('пропарсили слова и деньги')
time.sleep(n+30)

python3_command = "python2.7 make_classification.py"
process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

send_to_telegram ('сделали классификацию')
time.sleep(n+30)

python3_command = "python2.7 search_promiss.py"
process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

send_to_telegram ('сделали все')
time.sleep(n+30)

process = subprocess.Popen("python /root/projects/proj_alfa_promiss/delete_duplicates.py",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()
errcode = process.returncode


print ('finished iteration')
