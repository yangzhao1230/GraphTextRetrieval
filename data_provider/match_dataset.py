import torch
from torch_geometric.data import Dataset

from utils.GraphAug import drop_nodes, permute_edges, subgraph, mask_nodes
from copy import deepcopy
import numpy as np
import os
import random
from transformers import BertTokenizer
from transformers import RobertaTokenizer, RobertaModel

class GINMatchDataset(Dataset):
    def __init__(self, root, args):
        super(GINMatchDataset, self).__init__(root)
        self.root = root
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        self.data_type = args.data_type
        # print(self.graph_name_list[:10])
        # print(self.text_name_list[:10])

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        text_path = os.path.join(self.root, 'text', text_name)

        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line = line.strip('\n')
            text_list.append(line)
            if count > 500:
                break

        if self.data_type == 'para': # paragraph-level
            text, mask = self.tokenizer_text(text_list[0])
        
        if self.data_type == 'sent': #random sentence
            sts = text_list[0].split('.')
            remove_list = []
            for st in (sts):
                if len(st.split(' ')) < 5: 
                    remove_list.append(st)
            remove_list = sorted(remove_list, key=len, reverse=False)
            for r in remove_list:
                if len(sts) > 1:
                    sts.remove(r)
            text_index = random.randint(0, len(sts)-1)
            text, mask = self.tokenizer_text(sts[text_index])

        return data_graph, text.squeeze(0), mask.squeeze(0)

    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask

class KVMatchDataset(Dataset):
    def __init__(self, args, mode):
        super(KVMatchDataset, self).__init__()
        self.text_max_len = args.text_max_len
        self.mode = mode 
        self.graph_name_list = os.listdir(f'data/kv_data/{mode}/graph')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(f'data/kv_data/{mode}/text')
        self.text_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        self.data_type = args.data_type
        # print(self.graph_name_list[:10])
        # print(self.text_name_list[:10])

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        graph_path = os.path.join(f'data/kv_data/{self.mode}/graph', graph_name)
        data_graph = torch.load(graph_path)

        text_path = os.path.join(f'data/kv_data/{self.mode}/text', text_name)

        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line = line.strip('\n')
            text_list.append(line)
            if count > 500:
                break

        if self.data_type == 'para': # paragraph-level
            text, mask = self.tokenizer_text(text_list[0])
        
        if self.data_type == 'sent': #random sentence
            sts = text_list[0].split('.')
            remove_list = []
            for st in (sts):
                if len(st.split(' ')) < 5: 
                    remove_list.append(st)
            remove_list = sorted(remove_list, key=len, reverse=False)
            for r in remove_list:
                if len(sts) > 1:
                    sts.remove(r)
            text_index = random.randint(0, len(sts)-1)
            text, mask = self.tokenizer_text(sts[text_index])

        return data_graph, text.squeeze(0), mask.squeeze(0)

    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask
    
class SmilesMatchDataset(Dataset):
    def __init__(self, root, args):
        super(SmilesMatchDataset, self).__init__(root)
        self.root = root
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root+'smiles/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        self.data_type = args.data_type
        # print(self.graph_name_list[:10])
        # print(self.text_name_list[:10])

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        graph_path = os.path.join(self.root, 'smiles', graph_name)
        graph_list = []
        count = 0
        for line in open(graph_path, 'r', encoding='utf-8'):
            count += 1
            line = line.strip('\n')
            graph_list.append(line)
            if count > 500:
                break

        text_path = os.path.join(self.root, 'text', text_name)

        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line = line.strip('\n')
            text_list.append(line)
            if count > 500:
                break

        if self.data_type == 'para': # paragraph-level
            smiles, smiles_mask = self.tokenizer_text(graph_list[0])
            text, mask = self.tokenizer_text(text_list[0])
        
        if self.data_type == 'sent': #random sentence
            sts = text_list[0].split('.')
            remove_list = []
            for st in (sts):
                if len(st.split(' ')) < 5: 
                    remove_list.append(st)
            remove_list = sorted(remove_list, key=len, reverse=False)
            for r in remove_list:
                if len(sts) > 1:
                    sts.remove(r)
            text_index = random.randint(0, len(sts)-1)
            text, mask = self.tokenizer_text(sts[text_index])

        return smiles.squeeze(0), smiles_mask.squeeze(0), text.squeeze(0), mask.squeeze(0)

    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask

class ChemBertMatchDataset(Dataset):
    def __init__(self, root, args):
        super(ChemBertMatchDataset, self).__init__(root)
        self.root = root
        self.smiles_max_len = args.smiles_max_len
        self.text_max_len = args.text_max_len
        self.smiles_name_list = os.listdir(root+'smiles/')
        self.smiles_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_tokenizer = RobertaTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
        self.text_tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        self.data_type = args.data_type

    def __len__(self):
        return len(self.smiles_name_list)

    def __getitem__(self, index):
        smiles_name, text_name = self.smiles_name_list[index], self.text_name_list[index]
        smiles_path = os.path.join(self.root, 'smiles', smiles_name)
        with open(smiles_path, 'r', encoding='utf-8') as f_smiles:
            lines = f_smiles.readlines()
            smiles = lines[0].strip('\n')
        sm, sm_mask = self.tokenizer_smiles(smiles)
        text_path = os.path.join(self.root, 'text', text_name)

        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line = line.strip('\n')
            text_list.append(line)
            if count > 500:
                break
        
        text = mask = None

        if self.data_type == 'para': # parasmiles-level
            text, mask = self.tokenizer_text(text_list[0])
        
        if self.data_type == 'sent': #random sentence
            sts = text_list[0].split('.')
            remove_list = []
            for st in (sts):
                if len(st.split(' ')) < 5: 
                    remove_list.append(st)
            remove_list = sorted(remove_list, key=len, reverse=False)
            for r in remove_list:
                if len(sts) > 1:
                    sts.remove(r)
            text_index = random.randint(0, len(sts)-1)
            text, mask = self.tokenizer_text(sts[text_index])

        return sm.squeeze(0), sm_mask.squeeze(0), text.squeeze(0), mask.squeeze(0)#, index

    def tokenizer_smiles(self, text):
        tokenizer = self.smiles_tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.smiles_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask

    def tokenizer_text(self, text):
        tokenizer = self.text_tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask

class GINMatchShuffleDataset(Dataset):
    def __init__(self, args, ids, scaf):
        super(GINMatchShuffleDataset, self).__init__()
        self.ids = ids
        self.scaf = scaf
        self.graph_aug = args.graph_aug
        self.text_max_len = args.text_max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        self.data_type = args.data_type

    def __len__(self):
        return len(self.scaf)

    def __getitem__(self, index):
        # graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        idx = self.scaf[index]
        graph_path = './data/kv_data/graph/graph' + '_' + f"{self.ids[idx]}.pt"
        text_path = './data/kv_data/text/text' + '_' + f"{self.ids[idx]}.txt"
        # load graph data
        data_graph = torch.load(graph_path)
        # load text data
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text_list.append(line)
            if count > 1:
                break
        
        text = mask = None

        if self.data_type == 'para': # paragraph-level
            text, mask = self.tokenizer_text(text_list[0])
        
        if self.data_type == 'sent': #random sentence
            sts = text_list[0].split('.')
            remove_list = []
            for st in (sts):
                if len(st.split(' ')) < 5: 
                    remove_list.append(st)
            remove_list = sorted(remove_list, key=len, reverse=False)
            for r in remove_list:
                if len(sts) > 1:
                    sts.remove(r)
            text_index = random.randint(0, len(sts)-1)
            text, mask = self.tokenizer_text(sts[text_index])
        
        return data_graph, text.squeeze(0), mask.squeeze(0)#, index

    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask
    
class GINMatchScaffoldDataset(Dataset):
    def __init__(self, args, ids, scaf):
        super(GINMatchScaffoldDataset, self).__init__()
        # pubchem id in scaf
        self.scaf = scaf
        self.text_max_len = args.text_max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        self.data_type = args.data_type

        self.data_list = []
        for idx in self.scaf:
            file_id = ids[idx]
            graph_path = './data/phy_data/graph/graph' + '_' + f"{file_id}.pt"
            text_path = './data/phy_data/text/text' + '_' + f"{file_id}.txt"
            
            # load graph_data
            data_graph = torch.load(graph_path)
            # load text data
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                line.strip('\n')
                text_list.append(line)
                if count > 1:
                    break

            if self.data_type == 'para': # paragraph-level
                text, mask = self.tokenizer_text(text_list[0])
            
            if self.data_type == 'sent': #random sentence
                sts = text_list[0].split('.')
                remove_list = []
                for st in (sts):
                    if len(st.split(' ')) < 5: 
                        remove_list.append(st)
                remove_list = sorted(remove_list, key=len, reverse=False)
                for r in remove_list:
                    if len(sts) > 1:
                        sts.remove(r)
                text_index = random.randint(0, len(sts)-1)
                text, mask = self.tokenizer_text(sts[text_index])

            self.data_list.append((data_graph, text.squeeze(0), mask.squeeze(0)))

    def __len__(self):
        return len(self.scaf)

    def __getitem__(self, index):
        return self.data_list[index]

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask

class ChemBertMatchShuffleDataset(Dataset):
    def __init__(self, args, ids, scaf):
        super(ChemBertMatchShuffleDataset, self).__init__()
        self.ids = ids
        self.scaf = scaf
        self.smiles_max_len = args.smiles_max_len
        self.text_max_len = args.text_max_len
        # self.smiles_name_list = os.listdir(root+'smiles/')
        # self.smiles_name_list.sort()
        # self.text_name_list = os.listdir(root+'text/')
        # self.text_name_list.sort()
        self.smiles_tokenizer = RobertaTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
        self.text_tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        self.data_type = args.data_type

    def __len__(self):
        return len(self.scaf)

    def __getitem__(self, index):
        idx = self.scaf[index]
        smiles_path = './data/phy_data/smiles/smiles' + '_' + f"{self.ids[idx]}.txt"
        text_path = './data/phy_data/text/text' + '_' + f"{self.ids[idx]}.txt"

        for line in open(smiles_path, 'r', encoding='utf-8'):
            smiles = line.strip('\n')
            break
        
        sm, sm_mask = self.tokenizer_smiles(smiles)
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text_list.append(line)
            if count > 500:
                break
        
        text = mask = None

        if self.data_type == 'para': # parasmiles-level
            text, mask = self.tokenizer_text(text_list[0])
        
        if self.data_type == 'sent': #random sentence
            sts = text_list[0].split('.')
            remove_list = []
            for st in (sts):
                if len(st.split(' ')) < 5: 
                    remove_list.append(st)
            remove_list = sorted(remove_list, key=len, reverse=False)
            for r in remove_list:
                if len(sts) > 1:
                    sts.remove(r)
            text_index = random.randint(0, len(sts)-1)
            text, mask = self.tokenizer_text(sts[text_index])

        return sm.squeeze(0), sm_mask.squeeze(0), text.squeeze(0), mask.squeeze(0)#, index

    def tokenizer_smiles(self, text):
        tokenizer = self.smiles_tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.smiles_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask

    def tokenizer_text(self, text):
        tokenizer = self.text_tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask