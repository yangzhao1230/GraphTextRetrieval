import torch
import torch.nn as nn
from model.chembert import SmilesEncoder
from model.bert import TextEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim


class ChemBertSimclr(pl.LightningModule):
    def __init__(
            self,
            temperature=0.1,
            chembert_hidden_dim=384,
            # gin_num_layers,
            # drop_ratio,
            # smiles_pooling,
            smiles_self=False,
            bert_hidden_dim=768,
            bert_pretrain=True,
            projection_dim=256,
            lr=0.0001,
            weight_decay=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.temperature = temperature
        # self.gin_hidden_dim = gin_hidden_dim
        # self.gin_num_layers = gin_num_layers
        # self.drop_ratio = drop_ratio
        # self.smiles_pooling = smiles_pooling
        self.smiles_self = smiles_self
        self.chembert_hidden_dim = chembert_hidden_dim

        self.bert_hidden_dim = bert_hidden_dim
        self.bert_pretrain = bert_pretrain

        self.projection_dim = projection_dim

        self.lr = lr
        self.weight_decay = weight_decay

        self.smiles_encoder = SmilesEncoder()
        # print(self.smiles_encoder.state_dict().keys())
        # ckpt = torch.load('gin_pretrained/smilescl_80.pth')
        # print(ckpt.keys())
        # missing_keys, unexpected_keys = self.smiles_encoder.load_state_dict(ckpt, strict=False)
        # print(missing_keys)
        # print(unexpected_keys)

        if self.bert_pretrain:
            self.text_encoder = TextEncoder(pretrained=False)
        else:
            self.text_encoder = TextEncoder(pretrained=True)
            
        if self.bert_pretrain:
            print("bert load kvplm")
            ckpt = torch.load('kvplm_pretrained/ckpt_KV_1.pt')
            if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in ckpt:
                pretrained_dict = {"main_model."+k[20:]: v for k, v in ckpt.items()}
            elif 'bert.embeddings.word_embeddings.weight' in ckpt:
                pretrained_dict = {"main_model."+k[5:]: v for k, v in ckpt.items()}
            else:
                pretrained_dict = {"main_model."+k[12:]: v for k, v in ckpt.items()}
            # print(pretrained_dict.keys())
            # print(self.text_encoder.state_dict().keys())
            self.text_encoder.load_state_dict(pretrained_dict, strict=False)
            # missing_keys, unexpected_keys = self.text_encoder.load_state_dict(pretrained_dict, strict=False)
            # print(missing_keys)
            # print(unexpected_keys)
        # self.feature_extractor.freeze()

        self.smiles_proj_head = nn.Sequential(
          nn.Linear(self.chembert_hidden_dim, self.chembert_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.chembert_hidden_dim, self.projection_dim)
        )
        self.text_proj_head = nn.Sequential(
          nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.bert_hidden_dim, self.projection_dim)
        )

    def forward(self, features_smiles, features_text):
        batch_size = features_smiles.size(0)

        # normalized features
        features_smiles = F.normalize(features_smiles, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        logits_per_smiles = features_smiles @ features_text.t() / self.temperature
        logits_per_text = logits_per_smiles.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_smiles = F.cross_entropy(logits_per_smiles, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_smiles + loss_text) / 2

        return logits_per_smiles, logits_per_text, loss

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        sm, sm_mask, text1, mask1, text2, mask2 = batch

        smiles1_rep = self.smiles_encoder(sm, sm_mask)
        smiles1_rep = self.smiles_proj_head(smiles1_rep)

        smiles2_rep = smiles1_rep.clone()
        # smiles2_rep = self.smiles_encoder(sm2, sm_mask2)
        # smiles2_rep = self.smiles_proj_head(smiles2_rep)

        text1_rep = self.text_encoder(text1, mask1)
        text1_rep = self.text_proj_head(text1_rep)

        text2_rep = self.text_encoder(text2, mask2)
        text2_rep = self.text_proj_head(text2_rep)

        _, _, loss11 = self.forward(smiles1_rep, text1_rep)
        _, _, loss12 = self.forward(smiles1_rep, text2_rep)
        _, _, loss21 = self.forward(smiles2_rep, text1_rep)
        _, _, loss22 = self.forward(smiles2_rep, text2_rep)

        if self.smiles_self:
            _, _, loss_smiles_self = self.forward(smiles1_rep, smiles2_rep)
            loss = (loss11 + loss12 + loss21 + loss22 + loss_smiles_self) / 5.0
            # loss = (loss11 + loss12 + loss21 + loss22 + loss_smiles_self) / 4.0 + loss_smiles_self * 0.00025
        else:
            loss = (loss11 + loss12 + loss21 + loss22) / 4.0

        self.log("train_loss", loss)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
        # GIN
        # parser.add_argument('--gin_hidden_dim', type=int, default=300)
        # parser.add_argument('--gin_num_layers', type=int, default=5)
        # parser.add_argument('--drop_ratio', type=float, default=0.0)
        # parser.add_argument('--smiles_pooling', type=str, default='sum')
        parser.add_argument('--smiles_self', action='store_true', help='use smiles self-supervise or not', default=False)
        parser.add_argument('--chembert_hidden_dim', type=int, default=384, help='')
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_pretrain', action='store_false', default=True)
        parser.add_argument('--projection_dim', type=int, default=256)
        # optimization
        parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
        return parent_parser

