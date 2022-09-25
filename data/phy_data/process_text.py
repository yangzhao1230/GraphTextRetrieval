import os
import pandas as pd
import numpy as np
import json
import requests
import time
from tqdm import tqdm
import re

def update_text(cid):
    print(cid)
    flag = False
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/'
    req = requests.get(url)
    proper_json = json.loads(req.text)
    Section = proper_json['Record']['Section']
    for item in Section:
        if item['TOCHeading'] == 'Names and Identifiers':
            #Information = item['Section'][0]['Information']
            Section = item['Section']
            for item in Section:
                if item['TOCHeading'] == 'Record Description':
                    Information = item['Information']
                    Value = Information[0]['Value']
                    text = Value['StringWithMarkup'][0]['String']
                    flag = True
                    with open(f'./text/text_{cid}.txt', 'w', encoding='utf-8') as f:
                        f.writelines(text)
    print(flag)
cnt = 0

text_name_list = os.listdir('text/')
for text_name in text_name_list:
    with open(os.path.join('text', text_name), 'r', encoding='utf-8') as f:
            try:
                text_list = f.readlines()
            except:
                cnt = cnt + 1
                text_id = re.split('[_.]',text_name)[1]
                text_id = int(text_id)
                update_text(text_id)