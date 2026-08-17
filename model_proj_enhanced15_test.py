import dgl
import math
import torch
import torch as th
import torch.nn as nn
from torch.nn import init
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F

from utils import get_activation, to_etype_name
from torch.nn.parameter import Parameter

th.set_printoptions(profile="full")

#################################################
############### 改动2-1：给DP， PS层数1层会怎么样？TGCN搞一个E_layer层数 ###################
############### 改动2-2：如果PPI多几层/少几层会如何？，GCN搞3层 ###################
#################################################

# class GCN(nn.Module):
#     def __init__(self, features, nhid, nhid2, dropout):
#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolution(features, nhid)
#         self.gc2 = GraphConvolution(nhid, nhid2)
#
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return x
#
# class GCN_3layer(nn.Module):
#     def __init__(self, features, nhid, nhid2, nhid3, dropout):
#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolution(features, nhid)
#         self.gc2 = GraphConvolution(nhid, nhid2)
#         self.gc3 = GraphConvolution(nhid2, nhid3)
#
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc3(x, adj)
#         return x

class GCN(nn.Module):
    def __init__(self, in_features, nhid, nhid2, dropout=0.5, n_layers=2):
        """
        通用多层GCN模型
        参数：
            in_features: 输入特征维度
            hidden_dim:  每层隐藏单元数（可固定相同）
            out_features: 输出特征维度
            n_layers:     图卷积层数（>=2）
            dropout:      dropout比例
        """
        super(GCN, self).__init__()

        self.n_layers = n_layers
        self.dropout = dropout

        layers = []

        # 第一层
        layers.append(GraphConvolution(in_features, nhid))

        # 中间层
        for _ in range(n_layers - 2):
            layers.append(GraphConvolution(nhid, nhid))

        # 最后一层
        layers.append(GraphConvolution(nhid, nhid2))

        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj):
        for i, gc in enumerate(self.layers):
            x = gc(x, adj)
            # 最后一层不激活/不dropout
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x


class FGCN(nn.Module):
    def __init__(self, fdim_drug, fdim_disease, fdim_protein, nhid1, nhid2, dropout):
        super(FGCN, self).__init__()
        self.FGCN1 = GCN(fdim_drug, nhid1, nhid2, dropout)
        self.FGCN2 = GCN(fdim_disease, nhid1, nhid2, dropout)
        self.FGCN3 = GCN(fdim_protein, nhid1, nhid2, n_layers=3, dropout=dropout)
        self.pro_scgpt_proj = nn.Linear(512, fdim_protein, bias=False)  # 521 -> N_pro
        self.pro_view_gate = ViewGate(dim=fdim_protein, dropout=dropout)

        self.dropout = dropout

    def fuse_protein_feat(self, protein_sim_feat, protein_scgpt_feat):
        z_sim = protein_sim_feat  # [N_pro, N_pro]
        z_sc = self.pro_scgpt_proj(protein_scgpt_feat.detach())  # [N_pro, N_pro]
        z_fused = self.pro_view_gate(z_sim, z_sc)  # [N_pro, N_pro]
        return z_fused

    def forward(self, drug_graph, drug_sim_feat, dis_graph, disease_sim_feat, pro_graph, protein_sim_feat, protein_scgpt_feat):
        emb1 = self.FGCN1(drug_sim_feat, drug_graph)
        emb2 = self.FGCN2(disease_sim_feat, dis_graph)
        protein_sim_feat_pro = self.fuse_protein_feat(protein_sim_feat, protein_scgpt_feat)
        emb3 = self.FGCN3(protein_sim_feat_pro, pro_graph)

        return emb1, emb2, emb3


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = th.mm(input, self.weight)
        output = th.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),  # in_size=75
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z1, z2):
        z = torch.stack([z1, z2], dim=1)  # [B,2,D]
        w = self.project(z)
        beta = th.softmax(w, dim=1)  # [drug/dis_num, 2, 1]
        return (beta * z).sum(1)


class GCMCGraphConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats  # 909
        self._out_feats = out_feats  # 600
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        # init.xavier_uniform_(self.att)

    def forward(self, graph, feat, weight=None, Two_Stage=False):
        """Compute graph convolution.

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.
        dropout : torch.nn.Dropout, optional
            Optional external dropout layer.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat  # dst feature not used [drug or disease num , 3]
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise dgl.DGLError('External weight is provided while at the same time the'
                                       ' module has defined its own weight parameter. Please'
                                       ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)

            feat = feat * self.dropout(cj)
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_u(u='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = rst * ci

        return rst


class GCMCLayer(nn.Module):

    def __init__(self, rating_vals,  # [0, 1]
                 user_in_units,  # 909
                 movie_in_units,  # 909
                 msg_units,  # 1800
                 out_units,  # 75
                 dropout_rate=0.0,  # 0.3
                 agg='stack',  # 'sum'
                 agg_act=None,  # Tanh()
                 share_user_item_param=False,  # True
                 basis_units=4,
                 device=None, # True 4
                 protein_scgpt_feat: torch.Tensor = None,  # [N_pro, 512]
                 protein_start: int = None): # 第一个 protein 的 global_id:
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals  # [0, 1]
        self.agg = agg  # sum
        self.share_user_item_param = share_user_item_param  # True
        self.ufc = nn.Linear(msg_units, out_units)  # Linear(in_features=1800, out_features=75, bias=True)
        self.user_in_units = user_in_units  # 909
        self.msg_units = msg_units  # 1800
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
        if agg == 'stack':
            # divide the original msg unit size by number of rel_values to keep
            # the dimensionality
            assert msg_units % len(rating_vals) == 0
            msg_units = msg_units // len(rating_vals)

        msg_units = msg_units // 3  # 600
        self.msg_units = msg_units  # 600
        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = {}
        subConv = {}
        self.basis_units = basis_units  # 4
        self.att = nn.Parameter(th.randn(len(self.rating_vals), basis_units))  # [2, 4]

        self.basis = nn.Parameter(th.randn(basis_units, user_in_units, msg_units))  # [4, 909, 600]

        if protein_scgpt_feat is not None and protein_start is not None:
            # protein_scgpt_feat: [N_pro, 512]
            N_pro, sc_dim = protein_scgpt_feat.shape
            assert protein_start + N_pro <= user_in_units
            with torch.no_grad():
                # 一个简单的线性投影 512 -> msg_units（只用于初始化）
                proj = nn.Linear(sc_dim, msg_units, bias=False).to(device)
                # [N_pro, msg_units]
                pro_msg = proj(protein_scgpt_feat)
                # [basis_units, N_pro, msg_units]
                pro_msg_basis = pro_msg.unsqueeze(0).expand(self.basis_units, -1, -1)

                # 写到 global_id 对应区间
                self.basis.data[:, protein_start: protein_start + N_pro, :] = pro_msg_basis

        for i, rating in enumerate(rating_vals):
            # PyTorch parameter name can't contain "."
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            if share_user_item_param and user_in_units == movie_in_units:
                subConv[rating] = GCMCGraphConv(user_in_units,  # 909
                                                msg_units,  # 840
                                                weight=False,  # False
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(user_in_units,
                                                    msg_units,
                                                    weight=False,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
            else:
                self.W_r = None
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                msg_units,
                                                weight=True,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(movie_in_units,
                                                    msg_units,
                                                    weight=True,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        self.agg_act = get_activation(agg_act)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        """Put parameters into device except W_r

        Parameters
        ----------
        device : torch device
            Which device the parameters are put in.
        """
        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_user_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, entity1, entity2, entity1_feat=None, entity2_feat=None, Two_Stage=False):
        in_feats = {entity1: entity1_feat, entity2: entity2_feat}
        mod_args = {}
        self.W = th.matmul(self.att, self.basis.view(self.basis_units, -1))
        self.W = self.W.view(-1, self.user_in_units, self.msg_units)
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating

            mod_args[rating] = (self.W[i, :, :] if self.W_r is not None else None, Two_Stage)
            mod_args[rev_rating] = (self.W[i, :, :] if self.W_r is not None else None, Two_Stage)

        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        entity1_feat = out_feats[entity1]
        entity2_feat = out_feats[entity2]

        if in_feats[entity2].shape == entity2_feat.shape:
            ufeat = entity2_feat.view(entity2_feat.shape[0], -1)
            ifeat = entity1_feat.view(entity1_feat.shape[0], -1)

        entity1_feat = self.agg_act(entity1_feat)
        entity1_feat = self.dropout(entity1_feat)

        entity2_feat = self.agg_act(entity2_feat)
        entity2_feat = self.dropout(entity2_feat)

        entity1_feat = self.ifc(entity1_feat)
        entity2_feat = self.ufc(entity2_feat)

        return entity1_feat, entity2_feat

class NodeEmbedding(nn.Module):
    """
    全局节点 embedding：用 global_id 做索引(N_drug + N_dis + N_pro)。
    支持把 protein_scgpt_feat 投影后写入 protein 对应的行。
    """
    def __init__(self, num_nodes: int, emb_dim: int,
                 protein_scgpt_feat: torch.Tensor = None,   # (N_pro, 512)
                 protein_start: int = None,
                 device=None):
        super().__init__()
        self.num_embeddings = num_nodes
        self.emb = nn.Embedding(num_nodes, emb_dim)   # (N_drug + N_dis + N_pro, 75)
        self.device = device

        nn.init.xavier_uniform_(self.emb.weight)

        if protein_scgpt_feat is not None and protein_start is not None:
            protein_scgpt_feat = protein_scgpt_feat.to(device)
            N_pro, sc_dim = protein_scgpt_feat.shape
            assert protein_start + N_pro <= num_nodes

            # scgpt -> emb_dim 的投影（初始化用，也可以让它训练）
            self.proj = nn.Linear(sc_dim, emb_dim, bias=False).to(device)

            with torch.no_grad():
                pro_emb = self.proj(protein_scgpt_feat)  # [N_pro, emb_dim]
                self.emb.weight.data[protein_start:protein_start+N_pro] = pro_emb
        else:
            self.proj = None

    def forward(self, node_feat, tag):
        """
        node_feat:
          - [N,1] ：global_id
        """
        if node_feat is None:
            return None

        x = node_feat
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        if node_feat.shape[1] == 1:
            ids = x[:, 0].long()
            if self.device is not None:
                ids = ids.to(self.device)
            return self.emb(ids)  # [N, emb_dim]
        else:
            # x = x.to(self.device)
            return x

class GATGraphConv(nn.Module):
    """
    GAT
    """
    def __init__(self,
                 in_feats_src: int,
                 in_feats_dst: int,
                 out_feats: int,
                 num_heads: int = 1,
                 feat_drop: float = 0.0,
                 attn_drop: float = 0.0,
                 residual: bool = False,
                 allow_zero_in_degree: bool = True,
                 device=None):
        super().__init__()
        self.device = device

        # 每个 head 输出 out_feats，最终会 concat -> num_heads*out_feats
        self.gat = dglnn.GATConv(
            in_feats=(in_feats_src, in_feats_dst),
            out_feats=out_feats,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=residual,
            allow_zero_in_degree=allow_zero_in_degree
        )

    def forward(self, graph, feat):
        """
        graph: 该 etype 的子图（HeteroGraphConv 会传进来）
        return: [N_dst, num_heads*out_feats]
        """
        #  feat_src: [N_src, D_src]
        #  feat_dst: [N_dst, D_dst]
        # if isinstance(feat, tuple):
        #     feat_src, feat_dst = feat
        # else:
        #     feat_src = feat_dst = feat
        feat_src, feat_dst = feat

        with graph.local_scope():
            # if self.device is not None:
            #     graph = graph.to(self.device)
            #     feat_src = feat_src.to(self.device)
            #     feat_dst = feat_dst.to(self.device)
            out = self.gat(graph, (feat_src, feat_dst))  # [N_dst, heads, out_feats]
            out = out.flatten(1)  # [N_dst, heads*out_feats]
            return out

class GATLayer(nn.Module):
    """
    用 GATConv 做 message passing：
    - rating_vals/双向边（rev-）
    - agg_act + dropout
    - ifc/ufc 输出投影到 out_units
    """
    def __init__(self,
                 rating_vals,             # e.g. [1]
                 emb_dim,           # 280
                 # msg_units,               #  840，然后 //3 => 280；用 head*dim 对齐
                 out_units,               # 75
                 node_embed: NodeEmbedding,
                 dropout_rate=0.3,
                 agg='sum',
                 agg_act=None,
                 share_user_item_param=True,
                 num_heads=1,
                 attn_drop=0.0,
                 feat_drop=0.0,
                 residual=False,
                 device=None):
        super().__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.node_embed = node_embed
        self.share_user_item_param = share_user_item_param
        self.device = device

        self.dropout = nn.Dropout(dropout_rate)
        self.agg_act = (lambda x: x) if agg_act is None else agg_act

        # msg_units_after = msg_units // 3   # 280
        gat_out_per_head = max(emb_dim // num_heads, 1)  # 280
        self.gat_out_per_head = gat_out_per_head
        self.num_heads = num_heads
        self.msg_out_dim = gat_out_per_head * num_heads  # concat 后真实维度 280

        subConv = {}
        for rating in rating_vals:
            rname = to_etype_name(rating)
            rev_rname = f"rev-{rname}"

            # user/movie in_units 一样
            subConv[rname] = GATGraphConv(
                in_feats_src=emb_dim,
                in_feats_dst=emb_dim,
                out_feats=gat_out_per_head,
                num_heads=num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                residual=residual,
                allow_zero_in_degree=True,
                device=device
            )
            subConv[rev_rname] = GATGraphConv(
                in_feats_src=emb_dim,
                in_feats_dst=emb_dim,
                out_feats=gat_out_per_head,
                num_heads=num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                residual=residual,
                allow_zero_in_degree=True,
                device=device
            )

        # HeteroGraphConv 用 sum 聚合不同 etype 的输出（rating_vals=[1] 基本就一个）
        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)

        # 输出投影：把 msg_out_dim -> out_units
        self.ufc = nn.Linear(self.msg_out_dim, out_units)
        self.ifc = self.ufc if share_user_item_param else nn.Linear(self.msg_out_dim, out_units)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, entity1, entity2, entity1_feat=None, entity2_feat=None):
        """
        entity1: src node type name (e.g. 'drug')
        entity2: dst node type name (e.g. 'protein')
        """
        entity1_feat = self.node_embed(entity1_feat, entity1)
        entity2_feat = self.node_embed(entity2_feat, entity2)
        in_feats = {entity1: entity1_feat, entity2: entity2_feat}
        out_feats = self.conv(graph, in_feats)  # dict: {ntype: feat}

        entity1_h = out_feats[entity1]  # [N1, msg_out_dim]
        entity2_h = out_feats[entity2]  # [N2, msg_out_dim]

        entity1_h = self.dropout(self.agg_act(entity1_h))
        entity2_h = self.dropout(self.agg_act(entity2_h))

        entity1_h = self.ifc(entity1_h)
        entity2_h = self.ufc(entity2_h)
        return entity1_h, entity2_h


class MLPDecoder(nn.Module):
    def __init__(self,
                 in_units,
                 view_num=4,
                 dropout_rate=0.2):
        super(MLPDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

        self.lin1 = nn.Linear(view_num * in_units, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat
            graph.apply_edges(udf_u_mul_e)
            out = graph.edata['m']

            out = F.relu(self.lin1(out))
            out = self.dropout(out)

            out = F.relu(self.lin2(out))
            out = self.dropout(out)

            # out = self.sigmoid(self.lin3(out))
            out = self.lin3(out)
        return out




def udf_u_mul_e_norm(edges):
    return {'reg': edges.src['reg'] * edges.dst['ci']}
    # out_feats = edges.src['reg'].shape[1] // 3 return {'reg' : th.cat([edges.src['reg'][:, :out_feats] * edges.dst[
    # 'ci'], edges.src['reg'][:, out_feats:out_feats*2], edges.src['reg'][:, out_feats*2:]], 1)}


def udf_u_mul_e(edges):
    return {'m': th.cat([edges.src['h'], edges.dst['h']], 1)}
    # return {'m': (edges.src['h']) * (edges.dst['h'])}


def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix. A feat, B weight
    # feat size [313, 3] weight size [909, 600]
    if A is None:
        return B
    elif A.shape[1] == 3:
        if device is None:
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1)
        else:
            # return th.cat([B[A[:, 0].long()], B[A[:, 2].long()]], 1).to(device)  # only train one-hop
            # return th.cat([B[A[:, 0].long()], B[A[:, 1].long()]], 1).to(device)  # only train two-hop
            # return B[A[:, 0].long()].to(device)
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1).to(device)
    else:
        return A


# ---------- 投影, 对齐不同来源的分布 ----------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, dropout: float = 0.1, normalize: bool = False, residual: bool = False):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or out_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.BatchNorm1d(out_dim)
        self.normalize = normalize
        self.residual = residual and (in_dim == out_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        if self.residual:
            h = h + x
        h = self.ln(h)
        if self.normalize:
            h = F.normalize(h, p=2, dim=1)
        return h


# ---------- Gate 融合（样本级加权 + 残差兜底） ----------
class ViewGate(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        h = max(dim // 2, 1)
        self.g = nn.Sequential(
            nn.Linear(dim, h), nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(h, 1)
        )
    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        Z = torch.stack([z1, z2], dim=1)       # [B,2,D]
        scores = self.g(Z)                     # [B,2,1]
        alpha  = torch.softmax(scores, dim=1)  # [B,2,1]
        z = (alpha * Z).sum(1) + 0.5*(z1+z2)
        z = F.layer_norm(z, z.shape[-1:])
        return z

# ---------- 轻量 Cross-View 注意力,逐元素选通） ----------
class CrossViewAttention(nn.Module):
    def __init__(self, dim: int, hidden: int = None, dropout: float = 0.1):
        super().__init__()
        h = hidden or dim
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(2*dim, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, dim)
        )
    def forward(self, z_top: torch.Tensor, z_evi: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(z_top); K = self.Wk(z_evi); V = self.Wv(z_evi)
        alpha = torch.sigmoid((Q * K) / (Q.shape[-1] ** 0.5))  # [B,D]
        z_cv  = alpha * V
        z     = self.mlp(torch.cat([z_top, z_cv], dim=-1)) + z_top
        return F.layer_norm(z, z.shape[-1:])

# ---------- 融合器 ----------
class FusionManager(nn.Module):
    """
    mode: 'mean' | 'concat' | 'gate' | 'xattn' | 'xattn_gate'(限定intra)  -- inner and intra
    - mean       : (z1+z2)/2
    - concat     : [z1||z2] -> MLP -> D（带残差）
    - gate       : ViewGate（推荐 baseline）
    - xattn      : CrossView Attention
    - xattn_gate : CrossView 后再 Gate（推荐 inter_merge）
    """
    def __init__(self, dim: int, mode: str = 'gate', dropout: float = 0.1):
        super().__init__()
        self.mode = mode
        self.gate   = ViewGate(dim, dropout=dropout)
        self.xattn  = Attention(dim)
        self.concat = nn.Sequential(
            nn.Linear(2*dim, 2*dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2*dim, dim)
        )
    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        if self.mode == 'mean':
            return 0.5*(z1+z2)
        if self.mode == 'stack':
            z = self.concat(torch.cat([z1, z2], dim=-1)) + 0.5*(z1+z2)
            return F.layer_norm(z, z.shape[-1:])
        if self.mode == 'gate':
            return self.gate(z1, z2)  # (z, alpha)
        if self.mode == 'xattn':
            return self.xattn(z1, z2)
        if self.mode == 'xattn_gate':
            mid = self.xattn(z1, z2)
            return self.gate(mid, z2)
        raise ValueError(f'Unknown fusion mode: {self.mode}')



class Net(nn.Module):
    def __init__(self, args, protein_scgpt_feat):
        super(Net, self).__init__()
        self.layers = args.layers
        self.E_layers = args.E_layers
        self._act = get_activation(args.model_activation)
        self.protein_scgpt_feat = protein_scgpt_feat
        total_nodes = args.fdim_drug + args.fdim_disease + args.fdim_protein
        emb_dim = args.gcn_agg_units//3  # 280
        self.node_embed = NodeEmbedding(
            num_nodes=total_nodes,
            emb_dim=emb_dim,
            protein_scgpt_feat=self.protein_scgpt_feat,  # [N_pro, 512]
            protein_start=args.fdim_drug + args.fdim_disease,  # protein 的 global_id 起点
            device=args.device
        )

        ###########################################################################
        ########################### 修改：不同的图用不同的参数 #################################
        ###########################################################################
        self.TGCN_ds = nn.ModuleList()
        self.TGCN_ds.append(GCMCLayer(args.rating_vals,  # [0, 1]
                                   args.src_in_units,  # drug_num + disease_num + protein_num + 4
                                   args.dst_in_units,  # drug_num + disease_num + protein_num + 4
                                   args.gcn_agg_units,  # 840--- 调参
                                   args.gcn_out_units,  # 75
                                   args.dropout,  # 0.3
                                   args.gcn_agg_accum,  # sum
                                   agg_act=self._act,  # Tanh()
                                   share_user_item_param=args.share_param,  # True
                                   device=args.device))
        # drug-protein
        self.TGCN_dp = nn.ModuleList()
        self.TGCN_dp.append(
            GATLayer(
                rating_vals=[1],
                emb_dim=emb_dim,
                # msg_units=args.gcn_agg_units,
                out_units=args.gcn_out_units,
                node_embed=self.node_embed,
                dropout_rate=args.dropout,
                agg=args.gcn_agg_accum,
                agg_act=self._act,
                share_user_item_param=args.share_param,
                num_heads=1,
                feat_drop=0.0,
                attn_drop=0.0,
                residual=False,
                device=args.device
            )
        )

        # protein-disease
        self.TGCN_ps = nn.ModuleList()
        self.TGCN_ps.append(
            GATLayer(
                rating_vals=[1],
                emb_dim=emb_dim,
                # msg_units=args.gcn_agg_units,
                out_units=args.gcn_out_units,
                node_embed=self.node_embed,
                dropout_rate=args.dropout,
                agg=args.gcn_agg_accum,
                agg_act=self._act,
                share_user_item_param=args.share_param,
                num_heads=1,
                feat_drop=0.0,
                attn_drop=0.0,
                residual=False,
                device=args.device
            )
        )

        self.gcn_agg_accum = args.gcn_agg_accum  # sum
        self.rating_vals = args.rating_vals  # sum[0, 1]
        self.device = args.device
        self.gcn_agg_units = args.gcn_agg_units  # 840
        self.src_in_units = args.src_in_units  # drug_num + disease_num + protein_num + 4
        self.merge_type = args.merge_type

        # 对比学习前的MLP projection
        self.cl_drug_proj = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)
        # self.cl_drug_proj2 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)

        self.cl_dis_proj = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)
        # self.cl_dis_proj2 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)

        if args.merge_type == 'stack':
            self.cl_drug_proj3 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)
            self.cl_dis_proj3 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)

        self.cl_protein_proj = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)
        # self.cl_protein_proj2 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)
        # self.cl_protein_proj3 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)
        self.cl_pro_scgpt_proj = ProjectionHead(512, hidden_dim=args.gcn_out_units, out_dim=args.gcn_out_units,
                                                dropout=args.emb_dropout,
                                                normalize=True,   # 这里建议打开 L2 normalize，方便对齐
                                                residual=False)    # 维度不一样，残差就关掉  # 521 -> N_pro
        # inner fusion 前的 projection
        self.inner_drug_proj1 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)
        self.inner_drug_proj2 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)

        self.inner_disease_proj1 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)
        self.inner_disease_proj2 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)

        # intra fusion 前的projection

        self.intra_drug_proj1 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)
        self.intra_drug_proj2 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)

        self.intra_disease_proj1 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)
        self.intra_disease_proj2 = ProjectionHead(args.gcn_out_units, dropout=args.emb_dropout)

        ########## 聚合器--inner
        self.fuse_drug_inner = FusionManager(dim=args.gcn_out_units, mode=args.inner_concat_type, dropout=args.emb_dropout)
        self.fuse_disease_inner = FusionManager(dim=args.gcn_out_units, mode=args.inner_concat_type, dropout=args.emb_dropout)

        ########## 聚合器--intra
        self.fuse_drug_intra = FusionManager(dim=args.gcn_out_units, mode=args.intra_concat_type, dropout=args.emb_dropout)
        self.fuse_disease_intra = FusionManager(dim=args.gcn_out_units, mode=args.intra_concat_type, dropout=args.emb_dropout)

        for i in range(1, args.layers):
            if args.gcn_agg_accum == 'stack':
                gcn_out_units = args.gcn_out_units * len(args.rating_vals)
            else:
                gcn_out_units = args.gcn_out_units
            self.TGCN_ds.append(GCMCLayer(args.rating_vals,  # [0, 1]
                                       args.gcn_out_units,  # 75
                                       args.gcn_out_units,  # 75
                                       gcn_out_units,  # 75
                                       args.gcn_out_units,  # 75
                                       args.dropout,
                                       args.gcn_agg_accum,
                                       agg_act=self._act,
                                       share_user_item_param=args.share_param,
                                       device=args.device))
        for i in range(1, self.E_layers):
            if args.gcn_agg_accum == 'stack':
                gcn_out_units = args.gcn_out_units * len(args.rating_vals)
            else:
                gcn_out_units = args.gcn_out_units
            self.TGCN_dp.append(
                GATLayer(
                    rating_vals=[1],
                    emb_dim=args.gcn_out_units,
                    # msg_units=gcn_out_units,
                    out_units=args.gcn_out_units,
                    node_embed=self.node_embed,
                    dropout_rate=args.dropout,
                    agg=args.gcn_agg_accum,
                    agg_act=self._act,
                    share_user_item_param=args.share_param,
                    num_heads=1,
                    feat_drop=0.0,
                    attn_drop=0.0,
                    residual=False,
                    device=args.device
                )
            )
            self.TGCN_ps.append(
                GATLayer(
                    rating_vals=[1],
                    emb_dim=args.gcn_out_units,
                    # msg_units=gcn_out_units,
                    out_units=args.gcn_out_units,
                    node_embed=self.node_embed,
                    dropout_rate=args.dropout,
                    agg=args.gcn_agg_accum,
                    agg_act=self._act,
                    share_user_item_param=args.share_param,
                    num_heads=1,
                    feat_drop=0.0,
                    attn_drop=0.0,
                    residual=False,
                    device=args.device
                )
            )

        self.FGCN = FGCN(args.fdim_drug,
                         args.fdim_disease,
                         args.fdim_protein,
                         args.nhid1,
                         args.nhid2,
                         args.dropout)

        # self.attention_protein = Attention(args.gcn_out_units)
        # self.attention_drug = Attention(args.gcn_out_units)
        # self.attention_disease = Attention(args.gcn_out_units)
        if args.merge_type == 'stack':
            view_num = 6
        else:
            view_num = 2
        self.decoder = MLPDecoder(in_units=args.gcn_out_units, view_num=view_num)
        self.rating_vals = args.rating_vals

    def fuse_emb_enhanced(self, graph, emb_sim_out, emb_fuse_out, rev=True, skip=True):
        # 计算 drug 是否孤立（degree==0）
        # drug 的邻居数可以用反向 edge-type 的入度（rev-1 的 in_degrees）
        if rev:
            in_deg = graph['1'].in_degrees().to(self.device)
            out_deg = graph['rev-1'].out_degrees().to(self.device)
        else:
            in_deg = graph['rev-1'].in_degrees().to(self.device)
            out_deg = graph['1'].out_degrees().to(self.device)
        all_deg = in_deg + out_deg
        isolated_mask = (all_deg == 0)

        # 对孤立节点做“跳过聚合”或“残差保底”
        mask = isolated_mask.view(-1, 1).to(self.device)  # (N,1) 便于广播
        # 跳过聚合：直接使用 z1（drug_sim_out_proj）
        if skip:
            z_out = torch.where(mask, emb_sim_out, emb_fuse_out)
        else:
            residual_scale = 0.01
            z_out = torch.where(mask, emb_sim_out + residual_scale * emb_fuse_out, emb_fuse_out)
        return z_out


    def forward(self, enc_graph, dec_graph,
                drug_graph, drug_sim_feat, drug_feat,
                dis_graph, disease_sim_feat, dis_feat,
                pro_graph, pro_sim_feat,
                drug_evi_feat, dis_evi_feat, pro_feat,
                drug_pro_graph, pro_dis_graph,
                Two_Stage=False):

        # print(id(self.TGCN_dp[0].node_embed.emb.weight), id(self.TGCN_ps[0].node_embed.emb.weight))

        # Topology convolution operation
        drug_top_feat = drug_feat
        dis_top_feat = dis_feat
        drug_out, dis_out = None, None
        for i in range(0, self.layers):
            drug_o, dis_o = self.TGCN_ds[i](enc_graph, 'drug', 'disease', drug_top_feat, dis_top_feat, Two_Stage)
            if i == 0:
                drug_out = drug_o
                dis_out = dis_o

            else:
                drug_out += drug_o / float(i + 1)
                dis_out += dis_o / float(i + 1)

            drug_top_feat = drug_o
            dis_top_feat = dis_o

        ################# Evidence View下的 drug-pro, pro-dis 进行 异构 GCN的传播，目的是为了得到在该view下的 drug, dis,以及pro的表示
        drug_e_out, dis_e_out, pro_e_dout, pro_e_sout = None, None, None, None

        # pro_feat 首先融合pro_embedding
        pro_evi_dfeat = pro_feat
        pro_evi_sfeat = pro_feat

        for i in range(0, self.E_layers):
            drug_e_o, pro_e_do = self.TGCN_dp[i](drug_pro_graph, 'drug', 'protein', drug_evi_feat, pro_evi_dfeat)
            pro_e_so, dis_e_o = self.TGCN_ps[i](pro_dis_graph, 'protein', 'disease', pro_evi_sfeat, dis_evi_feat)
            if i == 0:
                drug_e_out = drug_e_o
                pro_e_dout = pro_e_do
                pro_e_sout = pro_e_so
                dis_e_out = dis_e_o
            else:
                drug_e_out = drug_e_out + drug_e_o / float(i + 1)
                pro_e_dout = pro_e_dout + pro_e_do / float(i + 1)
                pro_e_sout = pro_e_sout + pro_e_so / float(i + 1)
                dis_e_out = dis_e_out + dis_e_o / float(i + 1)
            drug_evi_feat = drug_e_o
            dis_evi_feat = dis_e_o
            pro_evi_dfeat = pro_e_do
            pro_evi_sfeat = pro_e_so

        # Feature convolution operation
        # pro_sim_feat 首先融合pro_embedding
        drug_sim_out, dis_sim_out, pro_sim_out = self.FGCN(drug_graph, drug_sim_feat,
                                                           dis_graph, disease_sim_feat,
                                                           pro_graph, pro_sim_feat, self.protein_scgpt_feat)


        ########################################################################
        #########################新增：在Evidence 下的融合方式，只有self.merge_type ！= 'stack'才会用到###############################
        # drug: drug_out, drug_e_out, drug_sim_out
        # disease: dis_out, dis_e_out, dis_sim_out
        # protein: pro_e_dout, pro_e_sout, pro_sim_out  -- 视角内不用聚合，视角外不用聚合，只对比学习用
        ########################################################################

        ########################################################################
        ######################## step1: evidence view 下的聚合-inner_concat_type ################################
        ########################################################################
        # 过一个投影层--》fuse
        drug_e_out_proj = self.inner_drug_proj1(drug_e_out)
        drug_sim_out_proj = self.inner_drug_proj2(drug_sim_out)
        drug_direct_out_proj = self.intra_drug_proj1(drug_out)

        dis_e_out_proj = self.inner_disease_proj1(dis_e_out)
        dis_sim_out_proj = self.inner_disease_proj2(dis_sim_out)
        dis_direct_out_proj = self.intra_disease_proj1(dis_out)

        if self.merge_type == 'stack':
            drug_feat = th.cat((drug_direct_out_proj, drug_sim_out_proj, drug_e_out_proj), 1)   # todo: 先concat, 最简单的先尝试
            dis_feat = th.cat((dis_direct_out_proj, dis_sim_out_proj, dis_e_out_proj), 1)
            pred_ratings = self.decoder(dec_graph, drug_feat, dis_feat)

            return pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out, drug_e_out, dis_e_out, pro_sim_out, pro_e_dout, pro_e_sout
        else:
            # inner 过一个投影层--》fuse
            drug_evidence_out = self.fuse_drug_inner(drug_sim_out_proj, drug_e_out_proj)
            dis_evidence_out = self.fuse_disease_inner(dis_sim_out_proj, dis_e_out_proj)

            # 处理孤立节点，针对DP和PS中的drug和dis节点
            drug_evidence_out = self.fuse_emb_enhanced(drug_pro_graph, drug_sim_out, drug_evidence_out, rev=False)
            dis_evidence_out = self.fuse_emb_enhanced(pro_dis_graph, dis_sim_out, dis_evidence_out)

            # intra聚合 -- 过一个投影层--》fuse
            drug_evidence_out_proj = self.intra_drug_proj2(drug_evidence_out)
            dis_evidence_out_proj = self.intra_disease_proj2(dis_evidence_out)

            if self.merge_type == 'fusion':    # intra fuse
                drug_feat = self.fuse_drug_intra(drug_direct_out_proj, drug_evidence_out_proj)
                dis_feat = self.fuse_disease_intra(dis_direct_out_proj, dis_evidence_out_proj)
                pred_ratings = self.decoder(dec_graph, drug_feat, dis_feat)
                return pred_ratings, drug_out, drug_evidence_out, dis_out, dis_evidence_out, pro_sim_out, pro_e_dout, pro_e_sout
            elif self.merge_type == 'fusion_cv':
                pred_ratings1 = self.decoder(dec_graph, drug_direct_out_proj, dis_direct_out_proj)   # todo: 先分别算不同View的分数，再加和
                pred_ratings2 = self.decoder(dec_graph, drug_evidence_out_proj, dis_evidence_out_proj)
                pred_ratings = pred_ratings1 + pred_ratings2
                return pred_ratings, drug_out, drug_evidence_out, dis_out, dis_evidence_out, pro_sim_out, pro_e_dout, pro_e_sout
            else:
                raise NotImplementedError
        # return pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out, drug_e_out, dis_e_out, pro_sim_out, pro_e_dout, pro_e_sout
