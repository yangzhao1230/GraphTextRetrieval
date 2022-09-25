import os.path as osp
from func_timeout import func_set_timeout
import func_timeout
import torch
from torch_geometric.data import Dataset, Data

from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict) 
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import pubchempy as pcp
import re
import os

def smiles2graph(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    
    atom_features_list = []

    if not mol:
        return None

    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)
    
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x

    return graph 

text_path = "./text"
graph_path = "./graph"
text_name_list = os.listdir(text_path)
graph_name_list = os.listdir(graph_path)
text_id_list = []
graph_id_list = []

for graph_name in graph_name_list:
    graph_id = re.split('[_.]',graph_name)[1]
    graph_id = int(graph_id)
    graph_id_list.append(graph_id)
for text_name in text_name_list:
    text_id = re.split('[_.]',text_name)[1]
    text_id = int(text_id)
    if text_id not in graph_id_list:
        text_id_list.append(text_id)

@func_set_timeout(20)
def f(cid):
    c = pcp.Compound.from_cid(cid)
    smiles = c.isomeric_smiles
    graph = smiles2graph(smiles)
    return graph

for cid in tqdm(text_id_list):
    print(cid)
    #line = line.strip('\n')
    
    graph = None
    try:
        graph = f(cid)
    except func_timeout.exceptions.FunctionTimedOut:
        print('timeout!')

    #index = index + 1
    pth = './graph'
    # if index <= 10500:
    #     pth = '/hy-tmp/KV-PLM/data/graph/train'
    # elif index <= 12000:
    #     pth = '/hy-tmp/KV-PLM/data/graph/dev'
    # else:
    #     pth = '/hy-tmp/KV-PLM/data/graph/test'

    if not graph:
        #data = None
        #torch.save(data, osp.join(pth, f'graph_{index}.pt'))
        continue

    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    torch.save(data, osp.join(pth, f'graph_{cid}.pt'))
