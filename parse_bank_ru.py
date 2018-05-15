
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

    bot = telegram.Bot(token='')
    chat_id = ''
    bot.send_message(chat_id=chat_id, text=text)
    time.sleep(2)

subprocess.check_output('docker run -i --rm -v ~/portia_projects:/app/data/projects:rw -v ~/portia_projects/output_data:/mnt:rw -p 9004:9004 scrapinghub/portia \
portiacrawl -s USER_AGENT="Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36" /app/data/projects/banki_ru bank -o /mnt/bank_ru_5000_all.json', shell=True)

send_to_telegram('банки ру пропарсили')

