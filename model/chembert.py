import torchvision.models as models
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import RobertaTokenizer, RobertaModel

# class BigModel(nn.Module):
#     def __init__(self, main_model):
#         super(BigModel, self).__init__()
#         self.main_model = main_model
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, tok, att, cud=True):
#         typ = torch.zeros(tok.shape).long()
#         if cud:
#             typ = typ.cuda()
#         pooled_output = self.main_model(tok, token_type_ids=typ, attention_mask=att)['pooler_output']
#         logits = self.dropout(pooled_output)
#         return logits


class SmilesEncoder(nn.Module):
    def __init__(self):
        super(SmilesEncoder, self).__init__()
        self.main_model = RobertaModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
        self.dropout = nn.Dropout(0.144)
        # self.hidden_size = self.main_model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        # device = input_ids.device
        # typ = torch.zeros(input_ids.shape).long().to(device)
        output = self.main_model(input_ids, attention_mask=attention_mask)['pooler_output']  # b,d
        logits = self.dropout(output)
        return logits


if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
    model = RobertaModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
    smiles = "C1=CC=CC=C1"
    inputs  = tokenizer(smiles)
    print(inputs)
    outputs = model(**inputs)['pooler_output']
    print(outputs.shape)