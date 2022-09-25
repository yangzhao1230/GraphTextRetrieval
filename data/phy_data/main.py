import pandas as pd
import numpy as np
import json
import requests
import time
from tqdm import tqdm

for cid in tqdm(range(54688, 54689)):
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


