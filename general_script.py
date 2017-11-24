import subprocess
import os
import telegram
import time
import pandas as pd
# import find_keywords

def send_to_telegram(text):

    """Send appropriate links to telegram channel"""

    bot = telegram.Bot(token='379005601:AAH1rv3ESXLWTXbn14gnCxW52eeKc4qnw50')
    # chat_id = -1001111732295
    chat_id = 169719023
    bot.send_message(chat_id=chat_id, text=text)
    time.sleep(10)


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
                        temp_text+=', '
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

        try:
            path = '/root/portia_projects/output_data/{}.json'.format(bank)
            print (path)
            os.remove(path)
            print ('ok')
        except OSError:
            pass

        subprocess.check_output('docker run -i --rm -v ~/portia_projects:/app/data/projects:rw -v ~/portia_projects/output_data:/mnt:rw -p 9003:9003 scrapinghub/portia \
        portiacrawl -s USER_AGENT="Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36" /app/data/projects/test {}'.format(bank) + ' -o /mnt/{}.json'.format(bank), shell=True)

    concated = read_and_concat(list_of_bank)
    concated.to_excel('../output/promiss/concated.xlsx', index=False)
    send_to_telegram('соединили файлы')


list_of_bank = ['alfabank.ru', 'sb','mb','uc','vtb','mcb','bnb','open', 'rf', 'ps']
# list_of_bank = ['mb','mcb','open']

for _ in range(100):
    run_parsing(list_of_bank)
    subprocess.Popen("python /root/projects/proj_alfa_promiss/find_keywords.py", shell=True)
    send_to_telegram('пропарсили ключевые слова')
    time.sleep(20)
    subprocess.Popen("python /root/projects/proj_alfa_promiss/parse_date_money.py", shell=True)
    send_to_telegram ('пропарсили слова и деньги')
    time.sleep(120)
    python3_command = "python2.7 make_classification.py"  # launch your python2 script using bash
    process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()  # receive output from the python2 script
    send_to_telegram ('сделали классификацию')
    time.sleep(20)
    python3_command = "python2.7 search_promiss.py"  # launch your python2 script using bash
    process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()  # receive output from the python2 script
    send_to_telegram ('сделали все')
    time.sleep(20)
    print ('finished iteration')

    time.sleep(7200)
