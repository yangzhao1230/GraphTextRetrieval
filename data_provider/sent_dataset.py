from torch_geometric.data import Dataset
import numpy as np
import os
from transformers import BertTokenizer

class GINSentDataset(Dataset):
    def __init__(self, root, args):
        super(GINSentDataset, self).__init__(root)
        self.root = root
        self.text_max_len = args.text_max_len
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        
        self.all_text = []
        self.all_mask = []
        self.cor = []
        cnt = 0
        #self.cor.append(cnt)
        for text_name in self.text_name_list:
            text_path = os.path.join(self.root, 'text', text_name)
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                line.strip('\n')
                text_list.append(line)
                if count > 500:
                    break

            sts = text_list[0].split('.')
            self.cor.append(cnt)
            for st in sts:
                if len(st.split(' ')) < 5:
                    continue
                text, mask = self.tokenizer_text(st)
                self.all_text.append(text)
                self.all_mask.append(mask)
                cnt+=1
        self.cor.append(cnt)
        np.save(f'{args.output_path}/cor.npy', self.cor)
        
    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, index):
        text = self.all_text[index]
        mask = self.all_mask[index]
        return text.squeeze(0), mask.squeeze(0)

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

class GINSentShuffleDataset(Dataset):
    def __init__(self, args, ids, scaf):
        super(GINSentShuffleDataset, self).__init__()
        self.ids = ids
        self.scaf = scaf
        self.text_max_len = args.text_max_len
        # self.text_name_list = os.listdir(root+'text/')
        # self.text_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        
        self.all_text = []
        self.all_mask = []
        self.cor = []
        cnt = 0
        #self.cor.append(cnt)
        for idx in self.scaf:
            id = ids[idx]
            # text_path = os.path.join(self.root, 'text', text_name)
            text_path = './data/phy_data/text/text' + '_' + f"{id}.txt"
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                line.strip('\n')
                text_list.append(line)
                if count > 500:
                    break

            sts = text_list[0].split('.')
            self.cor.append(cnt)
            for st in sts:
                if len(st.split(' ')) < 5:
                    continue
                text, mask = self.tokenizer_text(st)
                self.all_text.append(text)
                self.all_mask.append(mask)
                cnt+=1
        self.cor.append(cnt)
        np.save(f'{args.output_path}/cor.npy', self.cor)
        
    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, index):
        text = self.all_text[index]
        mask = self.all_mask[index]
        return text.squeeze(0), mask.squeeze(0)

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

class KVSentDataset(Dataset):
    def __init__(self, args, mode):
        super(KVSentDataset, self).__init__()
        self.text_max_len = args.text_max_len
        self.mode = mode
        self.graph_name_list = os.listdir(f'data/kv_data/{mode}/graph')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(f'data/kv_data/{mode}/text')
        self.text_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        self.data_type = args.data_type
        self.all_text = []
        self.all_mask = []
        self.cor = []
        cnt = 0
        #self.cor.append(cnt)
        for graph_name, text_name in zip(self.graph_name_list, self.text_name_list):

            text_path = os.path.join(f'data/kv_data/{self.mode}/text', text_name)
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                line.strip('\n')
                text_list.append(line)
                if count > 500:
                    break

            sts = text_list[0].split('.')
            self.cor.append(cnt)
            for st in sts:
                if len(st.split(' ')) < 5:
                    continue
                text, mask = self.tokenizer_text(st)
                self.all_text.append(text)
                self.all_mask.append(mask)
                cnt+=1
        self.cor.append(cnt)
        np.save(f'{args.output_path}/cor.npy', self.cor)
        
    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, index):
        text = self.all_text[index]
        mask = self.all_mask[index]
        return text.squeeze(0), mask.squeeze(0)

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

class GINSentScaffoldDataset(Dataset):
    def __init__(self, args, ids, scaf):
        super(GINSentScaffoldDataset, self).__init__()
        self.scaf = scaf
        self.text_max_len = args.text_max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        
        self.all_text = []
        self.all_mask = []
        self.cor = []
        cnt = 0
        for idx in self.scaf:
            file_id = ids[idx]
            text_path = './data/phy_data/text/text' + '_' + f"{file_id}.txt"
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                line.strip('\n')
                text_list.append(line)
                if count > 1:
                    break

            sts = text_list[0].split('.')
            self.cor.append(cnt)
            for st in sts:
                if len(st.split(' ')) < 5:
                    continue
                text, mask = self.tokenizer_text(st)
                self.all_text.append(text)
                self.all_mask.append(mask)
                cnt+=1
        self.cor.append(cnt)
        np.save(f'{args.output_path}/cor.npy', self.cor)
        
    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, index):
        text = self.all_text[index]
        mask = self.all_mask[index]
        return text.squeeze(0), mask.squeeze(0)

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