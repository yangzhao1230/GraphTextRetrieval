import numpy as np
from tqdm import tqdm
import pubchempy as pcp
import re
import os

text_path = "./text"
graph_path = "./graph"
smiles_path = './smiles'

text_name_list = os.listdir(text_path)
text_id_list = []
smiles_name_list = os.listdir(smiles_path)
smiles_id_list = []

for smiles_name in smiles_name_list:
    smiles_id = re.split('[_.]',smiles_name)[1]
    smiles_id = int(smiles_id)
    smiles_id_list.append(smiles_id)

for text_name in text_name_list:
    text_id = re.split('[_.]',text_name)[1]
    text_id = int(text_id)
    if text_id not in smiles_id_list:
        text_id_list.append(text_id)

for cid in tqdm(text_id_list):
    print(cid)
    c = pcp.Compound.from_cid(cid)
    smiles = c.isomeric_smiles
    with open(f'./smiles/smiles_{cid}.txt', 'w', encoding='utf-8') as f:
        f.writelines(smiles)