import argparse
import random
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from model.contrastive_gin_wo_smiles import GINSimclr
from data_provider.match_dataset import GINMatchShuffleDataset
from data_provider.sent_dataset import GINSentShuffleDataset
import torch_geometric
from optimization import BertAdam, warmup_linear
from torch.utils.data import RandomSampler
import os
import re 
from torch.utils.data import DataLoader 
import statistics
import logging

def prepare_model_and_optimizer(args, device):
    model = GINSimclr.load_from_checkpoint(args.init_checkpoint)
    if args.mode == 'linear':
        for p in model.graph_encoder.parameters():
            p.requires_grad = False
        for p in model.text_encoder.parameters():
            p.requires_grad = False

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]
    optimizer = BertAdam(
            optimizer_grouped_parameters,
            weight_decay=args.weight_decay,
            lr=args.lr,
            warmup=args.warmup,
            t_total=args.total_steps,
            )

    return model,optimizer

def Eval(model, dataloader, device, args):
    
    model.eval()
    with torch.no_grad():
        acc1 = 0
        acc2 = 0
        allcnt = 0
        graph_rep_total = None
        text_rep_total = None
        for batch in (dataloader):
            aug, text, mask = batch
            aug = aug.to(device)

            text = text.to(device)
            mask = mask.to(device)
            graph_rep = model.graph_encoder(aug)
            graph_rep = model.graph_proj_head(graph_rep)

            text_rep = model.text_encoder(text, mask)
            text_rep = model.text_proj_head(text_rep)

            scores1 = torch.cosine_similarity(graph_rep.unsqueeze(1).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), dim=-1)
            scores2 = torch.cosine_similarity(text_rep.unsqueeze(1).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), graph_rep.unsqueeze(0).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), dim=-1)

            argm1 = torch.argmax(scores1, axis=1)
            argm2 = torch.argmax(scores2, axis=1)

            acc1 += sum((argm1==torch.arange(argm1.shape[0]).to(device)).int()).item()
            acc2 += sum((argm2==torch.arange(argm2.shape[0]).to(device)).int()).item()

            allcnt += argm1.shape[0]

            if graph_rep_total is None or text_rep_total is None:
                graph_rep_total = graph_rep
                text_rep_total = text_rep
            else:
                graph_rep_total = torch.cat((graph_rep_total, graph_rep), axis=0)
                text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)

    np.save(f'{args.output_path}/graph_rep.npy', graph_rep_total.cpu())
    np.save(f'{args.output_path}/text_rep.npy', text_rep_total.cpu())

    return acc1/allcnt, acc2/allcnt

# get every sentence's rep
def CalSent(model, dataloader, device, args): 
    model.eval()
    with torch.no_grad():
        text_rep_total = None
        for batch in (dataloader):
            text, mask = batch
            text = text.to(device)
            mask = mask.to(device)
            text_rep = model.text_encoder(text, mask)
            text_rep = model.text_proj_head(text_rep)

            if text_rep_total is None:
                text_rep_total = text_rep
            else:
                text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)

    np.save(f'{args.output_path}/text_rep.npy', text_rep_total.cpu())

def Contra_Loss(logits_des, logits_smi, margin, device):
    scores = torch.cosine_similarity(logits_smi.unsqueeze(1).expand(logits_smi.shape[0], logits_smi.shape[0], logits_smi.shape[1]), logits_des.unsqueeze(0).expand(logits_des.shape[0], logits_des.shape[0], logits_des.shape[1]), dim=-1)
    diagonal = scores.diag().view(logits_smi.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    
    cost_des = (margin + scores - d1).clamp(min=0)
    cost_smi = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.to(device)
    cost_des = cost_des.masked_fill_(I, 0)
    cost_smi = cost_smi.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    #if self.max_violation:
    cost_des = cost_des.max(1)[0]
    cost_smi = cost_smi.max(0)[0]

    return cost_des.sum() + cost_smi.sum()

def main(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.device}')
    model, optimizer = prepare_model_and_optimizer(args, device)

    ids = []

    text_name_list = os.listdir("data/kv_data/text")
    for text_name in text_name_list:
        text_id = re.split('[_.]',text_name)[1]
        text_id = int(text_id)
        ids.append(text_id)
    ids.sort()
    seq = np.arange(len(ids))
    np.random.shuffle(seq)

    scaf = []
    k = int(len(seq)/10)
    scaf.append(seq[:7*k])
    scaf.append(seq[7*k:8*k])
    scaf.append(seq[8*k:])
    # sys.exit(0)

    TrainSet = GINMatchShuffleDataset(args,ids, scaf[0])
    DevSet = GINMatchShuffleDataset(args, ids, scaf[1])
    TestSet = GINMatchShuffleDataset(args, ids, scaf[2])
    train_sampler = RandomSampler(TrainSet)
    train_dataloader = torch_geometric.loader.DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=True)
    dev_dataloader = torch_geometric.loader.DataLoader(DevSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=True)
    test_dataloader = torch_geometric.loader.DataLoader(TestSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=False)#True
    global_step = 0
    tag = True
    best_acc = 0

    if args.mode != 'zeroshot': # finetune
        for epoch in range(args.epoch):
            if tag==False:
                break
            acc1, acc2 = Eval(model, dev_dataloader, device, args)
            print('Epoch:', epoch, ', DevAcc1:', acc1)
            print('Epoch:', epoch, ', DevAcc2:', acc2)
            if acc1>best_acc:
                best_acc = acc1
                torch.save(model.state_dict(), f'{args.output_path}/model.ckpt')
                print('Save checkpoint ', global_step)
            acc = 0
            allcnt = 0
            sumloss = 0
            model.train()
            for idx,batch in enumerate((train_dataloader)):
                aug, text, mask = batch
                aug.to(device)
                text = text.to(device)
                mask = mask.to(device)

                graph_rep = model.graph_encoder(aug)
                graph_rep = model.graph_proj_head(graph_rep)

                text_rep = model.text_encoder(text, mask)
                text_rep = model.text_proj_head(text_rep)

                loss = Contra_Loss(graph_rep, text_rep, args.margin, device)
                scores = text_rep.mm(graph_rep.t())
                argm = torch.argmax(scores, axis=1)
                acc += sum((argm==torch.arange(argm.shape[0]).to(device)).int()).item()
                allcnt += argm.shape[0]
                sumloss += loss.item()
                loss.backward()
                #if idx%4==1:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step>args.total_steps:
                    tag = False
                    break
            optimizer.step()
            optimizer.zero_grad()
            print('Epoch:', epoch, ', Acc:', acc/allcnt, ', Loss:', sumloss/allcnt)

        acc1, acc2 = Eval(model, dev_dataloader, device, args)
        print('Epoch:', args.epoch, ', DevAcc1:', acc1)
        print('Epoch:', args.epoch, ', DevAcc2:', acc2)
        if acc1>best_acc:
            best_acc = acc1
            torch.save(model.state_dict(), f'{args.output_path}/model.ckpt')
            print('Save checkpoint ', global_step)

        model.load_state_dict(torch.load(f'{args.output_path}/model.ckpt'))

    if args.data_type == 'para': # para-level

        acc1, acc2 = Eval(model, test_dataloader, device, args)
        print('Test Acc1:', round(acc1, 4))
        print('Test Acc2:', round(acc2, 4))
        graph_rep = torch.from_numpy(np.load(f'{args.output_path}/graph_rep.npy'))
        text_rep = torch.from_numpy(np.load(f'{args.output_path}/text_rep.npy'))
        graph_len = graph_rep.shape[0]
        text_len = text_rep.shape[0]
        score1 = torch.zeros(graph_len, graph_len)
        for i in range(graph_len):
            score1[i] = torch.cosine_similarity(graph_rep[i], text_rep, dim=-1)
        rec1 = []
        for i in range(graph_len):
            a,idx = torch.sort(score1[:,i])
            for j in range(graph_len):
                if idx[-1-j]==i:
                    rec1.append(j)
                    break
        rec_1 = sum( (np.array(rec1)<20).astype(int) ) / graph_len
        print(f'Rec@20 1: {round(rec_1, 4)}')
        rec1 = sum( (np.array(rec1)<20).astype(int) ) / graph_len

        score2 = torch.zeros(graph_len, graph_len)
        for i in range(graph_len):
            score2[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
        rec2 = []
        for i in range(graph_len):
            a,idx = torch.sort(score2[:,i])
            for j in range(graph_len):
                if idx[-1-j]==i:
                    rec2.append(j)
                    break
        rec_2 = sum( (np.array(rec2)<20).astype(int) ) / graph_len
        print(f'Rec@20 2: {round(rec_2, 4)}')
        rec2 = sum( (np.array(rec2)<20).astype(int) ) / graph_len
        return acc1, acc2, rec1, rec2
    
    else: #sent-level
        acc1, acc2 = Eval(model, test_dataloader, device, args)
        print(f"seed: {args.seed}")
        print('Test Acc1:', acc1)
        print('Test Acc2:', acc2)
        graph_rep = torch.from_numpy(np.load(f'{args.output_path}/graph_rep.npy'))
        SentSet = GINSentShuffleDataset(args, ids, scaf[2])
        sent_dataloader = DataLoader(SentSet, shuffle=False,
                                batch_size=args.batch_size,
                                num_workers=4, pin_memory=True, drop_last=False)#True
        CalSent(model, sent_dataloader, device, args)
        graph_rep = torch.from_numpy(np.load(f'{args.output_path}/graph_rep.npy'))
        text_rep = torch.from_numpy(np.load(f'{args.output_path}/text_rep.npy'))
        cor = np.load(f'{args.output_path}/cor.npy')
        
        graph_len = graph_rep.shape[0]
        text_len = text_rep.shape[0]

        score1 = torch.zeros(graph_len, graph_len)
        score2 = torch.zeros(graph_len, graph_len)

        for i in range(graph_len):
            score = torch.cosine_similarity(graph_rep[i], text_rep, dim=-1)
            for j in range(graph_len):
                total = 0
                for k in range(cor[j], cor[j+1]):
                    total+=(score[k]/(cor[j+1]-cor[j]))
                score1[i,j] = total
                #score1[i,j] = sum(score[cor[j]:cor[j+1]])/(cor[j+1]-cor[j])
        rec1 = []
        for i in range(graph_len):
            a,idx = torch.sort(score1[:,i])
            for j in range(graph_len):
                if idx[-1-j]==i:
                    rec1.append(j)
                    break
        print(f'Rec@20 1: {sum( (np.array(rec1)<20).astype(int) ) / graph_len}')
        rec1 = sum( (np.array(rec1)<20).astype(int) ) / graph_len

        score_tmp = torch.zeros(text_len, graph_len)
        for i in range(text_len):
            score_tmp[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
        score_tmp = torch.t(score_tmp)

        for i in range(graph_len):
            for j in range(graph_len):
                total = 0
                for k in range(cor[j], cor[j+1]):
                    total+=(score_tmp[i][k]/(cor[j+1]-cor[j]))
                score2[i,j] = total
                #score2[i,j] = sum(score_tmp[i][cor[j]:cor[j+1]])/(cor[j+1]-cor[j])
        score2 = torch.t(score2)

        rec2 = []
        for i in range(graph_len):
            a,idx = torch.sort(score2[:,i])
            for j in range(graph_len):
                if idx[-1-j]==i:
                    rec2.append(j)
                    break
        print(f'Rec@20 2: {sum( (np.array(rec2)<20).astype(int) ) / graph_len}')
        rec2 = sum( (np.array(rec2)<20).astype(int) ) / graph_len
        return acc1, acc2, rec1, rec2            

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--device", default="0", type=str,)
    parser.add_argument("--init_checkpoint", required=True, type=str,)
    parser.add_argument("--output_path", default='temp_path', type=str,)
    parser.add_argument('--data_type', choices=['sent', 'para'], required=True, help='Select data type: sent or para')
    parser.add_argument('--mode', choices=['zeroshot', 'finetune', 'linear'], required=True, help='Select mode: zeroshot, finetune, or linear')
    parser.add_argument("--weight_decay", default=0, type=float,)
    parser.add_argument("--lr", default=5e-5, type=float,)#4
    parser.add_argument("--warmup", default=0.2, type=float,)
    parser.add_argument("--total_steps", default=5000, type=int,)#3000
    parser.add_argument("--batch_size", default=64, type=int,)
    parser.add_argument("--epoch", default=30, type=int,)
    parser.add_argument("--seed", default=73, type=int,)#73 99 108
    parser.add_argument("--graph_aug", default='noaug', type=str,)
    parser.add_argument("--text_max_len", default=128, type=int,)
    parser.add_argument("--margin", default=0.2, type=int,)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    acc1_values = []
    acc2_values = []
    rec1_values = []
    rec2_values = []
    for seed in [73, 99, 108]:
        args.seed = seed
        print(f'seed:{args.seed}')
        acc1, acc2, rec1, rec2 = main(args)
        acc1_values.append(acc1)
        acc2_values.append(acc2)
        rec1_values.append(rec1)
        rec2_values.append(rec2)

    acc1_mean = statistics.mean(acc1_values)
    acc1_stddev = statistics.stdev(acc1_values) 
    acc2_mean = statistics.mean(acc2_values)
    acc2_stddev = statistics.stdev(acc2_values)
    rec1_mean = statistics.mean(rec1_values)
    rec1_stddev = statistics.stdev(rec1_values) 
    rec2_mean = statistics.mean(rec2_values)
    rec2_stddev = statistics.stdev(rec2_values) 

    # import pdb;pdb.set_trace()
    logging.basicConfig(filename=f'./logs/mode_{args.mode}_data_type_{args.data_type}_logs.txt', level=logging.INFO, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('*'* 50)
    logging.info(args.init_checkpoint)
    # logging.info(f'values:{values}')
    
    logging.info(f'acc1_mean: {acc1_mean:.4f}')
    logging.info(f'acc1_stddev: {acc1_stddev:.4f}')
    logging.info(f'acc2_mean: {acc2_mean:.4f}')
    logging.info(f'acc2_stddev: {acc2_stddev:.4f}')
    logging.info(f'rec1_mean: {rec1_mean:.4f}')
    logging.info(f'rec1_stddev: {rec1_stddev:.4f}')
    logging.info(f'rec2_mean: {rec2_mean:.4f}')
    logging.info(f'rec2_stddev: {rec2_stddev:.4f}')
    logging.info('*'* 50)