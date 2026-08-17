#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/10/29 21:16
# @Author  : Wenjing
# @File    : GCL_layer.py
# @Desc    : GCL based GCN layer


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter

class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features   # layer 0: 663
        self.out_features = out_features   # layer 0: 500
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))   # 663*500
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def node_dropout(x, drop_rate):
    if drop_rate == 0.0:
        return x
    mask = torch.rand(x.size(0)) > drop_rate
    x = x.clone()
    x[~mask] = 0
    return x

class GCN(nn.Module):
    def __init__(self, features, nhid, nhid2, dropout, drop_mode='OP', drop_rate=0.0):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(features, nhid)   # GCN(663,500）
        self.gc2 = GraphConvolution(nhid, nhid2)
        self.dropout = dropout
        self.drop_mode = drop_mode # 'OP', 'ED', 'ND'
        self.drop_rate = drop_rate

    def forward(self, x, adj):
        # x -- feature, 相似性分数； adj-- graph, 哪些节点之间边为1
        if self.drop_mode == 'ND':
            adj_dropped = adj
            x_dropped = node_dropout(x, self.drop_rate)    # 将相似性分数置为0
        else:
            adj_dropped = adj
            x_dropped = x

        x = F.relu(self.gc1(x_dropped, adj_dropped))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_dropped)
        return x

class FGCN(nn.Module):
    def __init__(self, fdim_drug, fdim_disease, nhid1, nhid2, dropout, drop_mode='OP', drop_rate=0.0):
        super(FGCN, self).__init__()
        self.FGCN1 = GCN(fdim_drug, nhid1, nhid2, dropout, drop_mode, drop_rate)   # fdim_drug： 663
        self.FGCN2 = GCN(fdim_disease, nhid1, nhid2, dropout, drop_mode, drop_rate)
        self.dropout = dropout
        self.drop_mode = drop_mode
        self.drop_rate = drop_rate

    def forward(self, drug_graph, drug_sim_feat, dis_graph, disease_sim_feat):
        emb1 = self.FGCN1(drug_sim_feat, drug_graph)
        emb2 = self.FGCN2(disease_sim_feat, dis_graph)
        return emb1, emb2




# Usage Example (in train loop, pseudo-code):
# fgcn = FGCN(..., drop_mode='edge', drop_rate=0.2)
# emb1_e, emb2_e = fgcn(drug_graph, drug_sim_feat, dis_graph, disease_sim_feat)
# fgcn2 = FGCN(..., drop_mode='node', drop_rate=0.2)
# emb1_n, emb2_n = fgcn2(drug_graph, drug_sim_feat, dis_graph, disease_sim_feat)
# loss_gcl = gcl_contrastive_loss(emb1_e, emb1_n) + gcl_contrastive_loss(emb2_e, emb2_n)
