
#!/usr/bin/env python3
import subprocess

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
    time.sleep(2)

subprocess.check_output('docker run -i --rm -v ~/portia_projects:/app/data/projects:rw -v ~/portia_projects/output_data:/mnt:rw -p 9004:9004 scrapinghub/portia \
portiacrawl -s USER_AGENT="Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36" /app/data/projects/banki_ru bank -o /mnt/bank_ru_5000_all.json', shell=True)

send_to_telegram('йоу все пропарсили')

# python3_command = "python2.7 make_classification.py"  # launch your python2 script using bash
#
# process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()  # receive output from the python2 script


# import os
#
# os.system('python find_keywords.py')
#
# print ('finished iteration')
# import subprocess
# subprocess.Popen("python /root/projects/proj_alfa_promiss/parse_date_money.py", shell=True)
