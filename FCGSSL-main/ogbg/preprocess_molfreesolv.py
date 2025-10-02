import argparse

import numpy as np
import torch
from torch import optim as optim
from torch.utils.data import DataLoader

from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from ogb.graphproppred import Evaluator

from dataset_ogb import OGBDataset
from dataloader import collate_fn

from autoencoder import GraphAutoEncoder
from encoder import GraphEncoder
from evaluation import ogbg_evaluation
from utils import load_config, create_optimizer, create_schedule, set_random_seed

import os
import random
import torch
import torch.nn as nn
import numpy as np
import scipy
import scipy.stats as st

from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected, degree
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.utils import get_laplacian
from torch_geometric.transforms import ToUndirected
import scipy as sp
import scipy.sparse as sps
import time
from scipy.io import loadmat
from collections import Counter
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Actor
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import degree


np.set_printoptions(precision=2, floatmode='fixed')  # 'fixed' 强制显示 2 位小数

# kitlee
def get_origin_edge_lappacian(edge_index, num_nodes):

    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float)

    # 计算对称归一化项 D^{-1/2}
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # 计算每条边的归一化权重 -1 / sqrt(deg[i] * deg[j])
    norm = -deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return norm

# kitlee
def get_rebuild_edge_lappacian(e, u, edge_index):
    
    fe1 = u @ torch.diag(e)
    fe2 = u
    fe1 = fe1[edge_index[0]]
    fe2 = fe2[edge_index[1]]
    return torch.sum(torch.abs(fe1 * fe2), dim=1)
    # return torch.sum(fe1 * fe2, dim=1)


def rank_array(arr):
    # 获取排序后的索引
    sorted_indices = np.argsort(arr)
    # 创建排名数组
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(arr)) + 1  # 排名从1开始
    
    # 处理相同值的情况
    unique_values, first_indices = np.unique(arr, return_index=True)
    for val in unique_values:
        mask = arr == val
        ranks[mask] = ranks[mask].min()  # 相同值取最小排名
    
    return ranks


dataset = OGBDataset(root="./dataset", dataset="ogbg-molfreesolv")

# print(dataset)

# data = dataset[0]

# print(data)


# if data.is_directed():
#     print("is not undirected")
#     data.edge_index = to_undirected(data.edge_index)

x_dataset = []

for i in range(len(dataset)):

    data = dataset[i]

    index, attr = get_laplacian(data.edge_index, normalization='sym')
    L = to_scipy_sparse_matrix(index, attr)

    L = torch.FloatTensor(L.todense())
    e, u = torch.linalg.eigh(L)

    # e, u = scipy.sparse.linalg.eigsh(L, k=800, which='SA', tol=1e-5)
    data.e = torch.FloatTensor(e)
    data.u = torch.FloatTensor(u)

    data.num_nodes = len(e)

    # 获取总的
    bl = 1
    bs = int(len(e) / bl) + 1 
    rs = []
    for i in range(bl):
        r = get_rebuild_edge_lappacian(e[bs*i:bs*(i+1)], u[:,bs*i:bs*(i+1)], data.edge_index).tolist()
        rs.append(r)
    rs = np.array(rs)  # 10 * e
    rs = rs.T  # e * 10
    rs_total = np.sum(rs, axis=1)

    # 获取每一项
    rs_freq = np.array([0 for _ in range(len(data.edge_index[0]))])
    rs_temp = np.array([0 for _ in range(len(data.edge_index[0]))])
    for i in range(len(e)):
        rp = get_rebuild_edge_lappacian(e[i:(i+1)], u[:,i:(i+1)], data.edge_index).tolist()

        rs_temp = rs_temp + rp
        rs_freq = rs_freq + rs_temp / rs_total

    edge_drop_pro = rs_freq / len(e)

    node_mask_pro = [0 for _ in range(data.num_nodes)]
    node_degree = [0 for _ in range(data.num_nodes)]
    for i in range(len(edge_drop_pro)):

        node_degree[data.edge_index[0][i]] += 1
        node_degree[data.edge_index[1][i]] += 1
        edp = edge_drop_pro[i]
        if edp > 0:
            node_mask_pro[data.edge_index[0][i]] += edp
            node_mask_pro[data.edge_index[1][i]] += edp
        

    node_mask_pro = np.array(node_mask_pro)
    node_degree = np.array(node_degree)
    node_mask_pro = node_mask_pro / node_degree
    edge_drop_pro = np.array(edge_drop_pro)

    node_mask_pro[np.isnan(node_mask_pro)] = 0         # 替换 NaN
    node_mask_pro[np.isinf(node_mask_pro)] = 0         # 替换 inf
    node_mask_pro[node_mask_pro < 0] = 0               # 替换负数

    edge_drop_pro[np.isnan(edge_drop_pro)] = 0         # 替换 NaN
    edge_drop_pro[np.isinf(edge_drop_pro)] = 0         # 替换 inf
    edge_drop_pro[edge_drop_pro < 0] = 0               # 替换负数00001


    x_data = {}

    x_data['edge_drop_pro_num'] = edge_drop_pro
    x_data['node_mask_pro_num'] = node_mask_pro

    x_data['edge_drop_pro_qua'] = rank_array(edge_drop_pro)
    x_data['node_mask_pro_qua'] = rank_array(node_mask_pro)

    x_dataset.append(x_data)

torch.save(x_dataset, '../dataset/ogbg-molfreesolv.pt')