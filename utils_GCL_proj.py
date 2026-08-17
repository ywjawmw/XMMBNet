#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/10/30 15:37
# @Author  : Wenjing
# @File    : utils_GCL.py
# @Desc    :  做edge dropout 和 node dropout, 分同质图和异质图

import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import csv

class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()

def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])


def torch_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) + \
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


def get_optimizer(opt):
    if opt == 'sgd':
        return optim.SGD
    elif opt == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError


def to_etype_name(rating):
    return str(rating).replace('.', '_')



def drop_edges_undirected_scipy(A_coo, drop_rate, keep_self_loop=False, rng=None):
    """
    在 SciPy COO 稀疏矩阵上对无向图成对丢边（保证对称）。
    A_coo: 未归一化的对称邻接（0/1），dtype=float32
    """
    if rng is None:
        rng = np.random.default_rng()
    A = A_coo.tocoo()
    row, col = A.row, A.col
    # 上三角（含自环）
    u = np.minimum(row, col)
    v = np.maximum(row, col)
    uv = np.stack([u, v], axis=1)
    # 去重到无向边集合
    uv_unique, idx = np.unique(uv, axis=0, return_inverse=True)
    E = uv_unique.shape[0]

    # 每对无向边采样一次；自环可选择强制保留
    keep = rng.random(E) > drop_rate
    if keep_self_loop:
        keep = np.where(uv_unique[:,0] == uv_unique[:,1], True, keep)

    uv_kept = uv_unique[keep]
    # 还原成对称边：非自环补反向，自环只保留一次
    nonself = uv_kept[uv_kept[:,0] != uv_kept[:,1]]
    selfonly = uv_kept[uv_kept[:,0] == uv_kept[:,1]]

    i = np.concatenate([nonself[:,0], nonself[:,1], selfonly[:,0]])
    j = np.concatenate([nonself[:,1], nonself[:,0], selfonly[:,1]])
    data = np.ones_like(i, dtype=A.dtype)

    N = A.shape[0]
    A_drop = sp.coo_matrix((data, (i, j)), shape=(N, N), dtype=A.dtype)
    return A_drop


def multi_positive_ntxent(views, tau: float = 0.2):
    """
    views: List[Tensor]，每个形状 [N, D]，例如 [z1, z2, z3]
           要求同一条样本在各视角的顺序对齐（第 i 行互为正例）。
    返回: 标量 loss
    """
    K = len(views)
    assert K >= 2, "需要至少两个视角"
    N, D = views[0].shape
    assert all(z.shape == (N, D) for z in views), "各视角张量形状需一致"

    Z = torch.cat(views, dim=0)                   # [K*N, D]
    Z = F.normalize(Z, dim=1)                     # 余弦相似
    S = (Z @ Z.T) / tau                           # [K*N, K*N]，未取 exp，用 logsumexp 做

    KN = K * N
    device = Z.device

    # 构造 “同样本”标签：0..N-1 重复 K 次
    labels = torch.arange(N, device=device).repeat(K)      # [K*N]
    same_sample = labels.unsqueeze(0) == labels.unsqueeze(1)  # [K*N, K*N] bool

    # 自身位置（对角线）不参与
    eye = torch.eye(KN, dtype=torch.bool, device=device)

    # 正样本掩码：同样本且非自身 -> 每行会有 K-1 个 True
    pos_mask = same_sample & (~eye)

    # 确保每行至少有一个正样本
    pos_count = pos_mask.sum(dim=1)
    if (pos_count == 0).any():
        raise RuntimeError("Each row must have at least one positive sample (check K and input shapes)")

    # 分子：对正样本子集做 logsumexp
    S_pos = S.masked_fill(~pos_mask, float('-inf'))
    lse_pos = torch.logsumexp(S_pos, dim=1)       # [K*N]

    # 分母：对“所有非自身”做 logsumexp（包含正样本和所有负样本）
    S_all = S.masked_fill(eye, float('-inf'))
    lse_all = torch.logsumexp(S_all, dim=1)       # [K*N]

    loss = -(lse_pos - lse_all).mean()
    return loss

def LOSS_multi(args, z1: torch.Tensor, z2: torch.Tensor, z3: None,
        mean: bool = True, batch_size: int = 0, flag: int = 0):
    input_tensor = [z1, z2] if z3 is None else [z1, z2, z3]
    loss= multi_positive_ntxent(input_tensor, args.tau)
    return loss

def LOSS_multi_2(args, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, z4: torch.Tensor,
        mean: bool = True, batch_size: int = 0, flag: int = 0):
    input_tensor = [z1, z2, z3, z4]
    loss= multi_positive_ntxent(input_tensor, args.tau)
    return loss


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(args, z1: torch.Tensor, z2: torch.Tensor, flag: int):
    f = lambda x: torch.exp(x / args.tau)
    refl_sim = f(args.intra * sim(z1, z1))  # torch.Size([663, 663])
    between_sim = f(args.inter * sim(z1, z2))  # z1 z2:torch.Size([663, 75])

    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


def LOSS(args, z1: torch.Tensor, z2: torch.Tensor,
        mean: bool = True, batch_size: int = 0, flag: int = 0):
    l1 = semi_loss(args, z1, z2, flag)
    l2 = semi_loss(args, z2, z1, flag)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()
    return ret
