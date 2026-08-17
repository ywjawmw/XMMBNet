import os
import dgl
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch as th

from utils import *
from sklearn.model_selection import KFold
from utils_GCL import *

# 处理drug-pro, pro-dis 两个异质图, 注意改动是 这里不再考虑负边
_paths = {
    'Gdataset': './name_data/drug_data/Gdataset',
    'Cdataset': './name_data/drug_data/Cdataset',
    'Gdataset_e': './name_data/drug_data/Gdataset_e',
    'Cdataset_e': './name_data/drug_data/Cdataset_e',
    'Adataset': './name_data/drug_data/Adataset',
    'lrssl': './name_data/drug_data',

}


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


class DrugDataLoader(object):
    def __init__(self,
                 args,
                 name,
                 device,
                 symm=True,
                 k=2):
        self._name = name
        self._device = device
        self._symm = symm
        self.num_neighbor = k
        self.ppi_radio = args.ppi_radio
        print("Starting processing {} ...".format(self._name))
        self._dir = os.path.join(_paths[self._name])
        self.cv_data_dict = self._load_drug_data(self._dir, self._name)


        self._generate_topoy_graph()
        if args.drop_type == "ED":
            self.drug_graph, self.disease_graph, self.protein_graph, self.drug_graph_enhanced, self.disease_graph_enhanced, self.protein_graph_enhanced = self._generate_feat_graph(args)
        else:
            self.drug_graph, self.disease_graph, self.protein_graph = self._generate_feat_graph(args)
        self._generate_feat()   # 构建不同节点的特征 3列
        self.possible_rel_values = self.values  # （0/1)

        #######构建 drug-pro, pro-dis 两个异质图
        self.drug_pro_graph = self._generate_evidence_graph(self.drug_pro_matrix, 'drug', 'protein', self.num_drug, self.num_protein)
        self.pro_dis_graph = self._generate_evidence_graph(self.pro_dis_matrix, 'protein', 'disease', self.num_protein, self._num_disease)

    def _load_drug_data(self, file_path, data_name):
        association_matrix = None
        if data_name in ['Gdataset', 'Cdataset', 'Adataset']:
            # data = sio.loadmat(file_path)
            # association_matrix = data['didr'].T
            # self.disease_sim_features = data['disease']
            # self.drug_sim_features = data['drug']
            ### 修改为从csv文件中读取的形式, 保证csv中的数据正确
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.csv'), index_col=0, delimiter=',')
            association_matrix = data.values
            self.disease_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.csv'), index_col=0, delimiter=',').values
            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.csv'), index_col=0, delimiter=',').values
            # 和 protein相关的三个图，均为全部值load
            self.protein_sim_features = pd.read_csv(
                os.path.join(file_path, 'pro_sim_enhanced.csv'), index_col=0, delimiter=',').values   # PPI的相似值
            self.drug_pro_matrix =  pd.read_csv(
                os.path.join(file_path, 'drug_pro_enhanced.csv'), index_col=0, delimiter=',').values   # 邻接矩阵 0/1
            self.pro_dis_matrix = pd.read_csv(
                os.path.join(file_path, 'pro_dis_enhanced.csv'), index_col=0, delimiter=',').values  # 邻接矩阵 0/1
            self.ppi_adj = pd.read_csv(
                os.path.join(file_path, f'ppi_adj_enhanced-{self.ppi_radio}.csv'), index_col=0, delimiter=',').values  # 邻接矩阵 0/1
            self.pro_embedding = torch.load(
                os.path.join(file_path, 'gene_emb.pt'))

        self._num_drug = association_matrix.shape[0]
        self._num_disease = association_matrix.shape[1]
        self._num_protein = self.protein_sim_features.shape[0]

        kfold = KFold(n_splits=10, shuffle=True, random_state=1024)
        pos_row, pos_col = np.nonzero(association_matrix)
        neg_row, neg_col = np.nonzero(1 - association_matrix)
        assert len(pos_row) + len(neg_row) == np.prod(association_matrix.shape)
        cv_num = 0
        cv_data = {}
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                kfold.split(neg_row)):
            train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
            train_pos_values = [1] * len(train_pos_edge[0])
            train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
            train_neg_values = [0] * len(train_neg_edge[0])

            test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
            test_pos_values = [1] * len(test_pos_edge[0])

            '''
            # test positive and test negative ration is 1:1
                test_neg_edge = np.stack([neg_row[test_neg_idx][0:len(test_pos_values)],
                                          neg_col[test_neg_idx][0:len(test_pos_values)]])

            '''

            test_neg_edge = np.stack([neg_row[test_neg_idx],
                                      neg_col[test_neg_idx]])
            test_neg_values = [0] * len(test_neg_edge[0])

            train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
            train_values = np.concatenate([train_pos_values, train_neg_values])
            test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
            test_values = np.concatenate([test_pos_values, test_neg_values])

            train_data = {
                'drug_id': train_edge[0],
                'disease_id': train_edge[1],
                'values': train_values
            }
            train_data_info = pd.DataFrame(train_data, index=None)

            test_data = {
                'drug_id': test_edge[0],
                'disease_id': test_edge[1],
                'values': test_values
            }
            test_data_info = pd.DataFrame(test_data, index=None)
            values = np.unique(train_values)
            cv_data[cv_num] = [train_data_info, test_data_info, values]
            cv_num += 1

        return cv_data

    def _generate_feat(self):
        ####################################################################
        ##################### 修改：不要protein，GCMC只给drug-disease用，特征维度应该是两类实体相加，同时预留3个特殊占位， 0-大家共有，1-drug edge, 2- disease edge #######################
        ####################################################################
        self.drug_feature_shape = (self.num_drug, self.num_drug + self.num_disease + 3) # (663, 1075)
        self.disease_feature_shape = (self.num_disease, self.num_drug + self.num_disease + 3) # (409, 1075)
        # self.protein_feature_shape = (self.num_protein, 1) # (993, 1)

        self.drug_feature = th.cat(
            [th.Tensor(list(range(3, self.num_drug + 3))).reshape(-1, 1), th.zeros([self.num_drug, 1]) + 1,
             th.zeros([self.num_drug, 1])], 1) # 第一列：4,5,..., 666； 第二列： 全1， 第三列： 全0

        self.disease_feature = th.cat(
            [th.Tensor(list(range(self.num_drug + 3, self.num_drug + self.num_disease + 3))).reshape(-1, 1),
             th.ones([self.num_disease, 1]) + 1, th.zeros([self.num_disease, 1])], 1)   # 第一列：667, 668,..., 1075； 第二列： 全2， 第三列： 全0
        ########## GAT 使用的id index
        self.drug_evi_feat = th.cat(
            [th.Tensor(list(range(0, self.num_drug))).reshape(-1, 1)], 1)  # 第一列：0-662；
        self.dis_evi_feat = th.cat(
            [th.Tensor(list(range(self.num_drug, self.num_drug + self.num_disease))).reshape(-1, 1)],
            1)  # 第一列：663-1071；
        self.pro_feat = th.cat(
            [th.Tensor(list(range(self.num_drug + self.num_disease,
                                  self.num_drug + self.num_disease + self.num_protein))).reshape(-1, 1)],
            1)  # 第一列：1072-；

    def _generate_topoy_graph(self):
        self.data_cv = {}
        for cv in range(0, 10):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]
            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_rel_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_rel_info = self.test_data
            self.possible_rel_values = self.values

            train_pairs, train_values = self._generate_pair_value(
                self.train_rel_info)
            test_pairs, test_values = self._generate_pair_value(self.test_rel_info)

            self.train_enc_graph = self._generate_enc_graph(train_pairs, train_values,
                                                            add_support=True)
            self.train_dec_graph = self._generate_dec_graph(train_pairs)
            self.train_truths = th.FloatTensor(train_values)

            self.test_enc_graph = self.train_enc_graph
            self.test_dec_graph = self._generate_dec_graph(test_pairs)
            self.test_truths = th.FloatTensor(test_values)
            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths]}
        return self.data_cv

    def _generate_evidence_graph(self, heterogeneous_matrix, entity1, entity2, entity1_num, entity2_num):
        ####################################################################
        ######################## 新增：处理drug-pro, pro-dis 两个异质图, 注意这里不再考虑负边 ############################
        ####################################################################
        pos_row, pos_col = np.nonzero(heterogeneous_matrix)
        # 只保留正例构造的rating pairs（用于 GCMC 传播）
        rating_pairs_pos = (pos_row.astype(np.int64), pos_col.astype(np.int64))
        rating_values_pos = np.array([1] * len(pos_row), dtype=np.float32)

        heterogeneous_graph = self._generate_heterogeneous_graph_no_neg(
            rating_pairs_pos, rating_values_pos, entity1, entity2, entity1_num, entity2_num, add_support=True
        )
        return heterogeneous_graph

    def _generate_heterogeneous_graph_no_neg(self, rating_pairs, rating_values, entity1, entity2, entity1_num, entity2_num,
                                      add_support=False):
        ####################################################################
        ######################## 新增：处理drug-pro, pro-dis 两个异质图, 不考虑负边 ############################
        ####################################################################
        data_dict = dict()
        num_nodes_dict = {entity1: entity1_num, entity2: entity2_num}
        rating_row, rating_col = rating_pairs
        for rating in [1]:
            ridx = np.where(
                rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                (entity1, str(rating), entity2): (rrow, rcol),
                (entity2, 'rev-%s' % str(rating), entity1): (rcol, rrow)
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            entity1_ci = []
            entity1_cj = []
            entity2_ci = []
            entity2_cj = []
            for r in [1]:
                r = to_etype_name(r)
                entity1_ci.append(graph['rev-%s' % r].in_degrees())
                entity2_ci.append(graph[r].in_degrees())
                if self._symm:
                    entity1_cj.append(graph[r].out_degrees())
                    entity2_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    entity1_cj.append(th.zeros((entity1_num,)))
                    entity2_cj.append(th.zeros((entity2_num,)))

            entity1_ci = _calc_norm(sum(entity1_ci))
            entity2_ci = _calc_norm(sum(entity2_ci))
            if self._symm:
                entity1_cj = _calc_norm(sum(entity1_cj))
                entity2_cj = _calc_norm(sum(entity2_cj))
            else:
                entity1_cj = th.ones(entity1_num, )
                entity2_cj = th.ones(entity2_num, )
            graph.nodes[entity1].data.update({'ci': entity1_ci, 'cj': entity1_cj})
            graph.nodes[entity2].data.update({'ci': entity2_ci, 'cj': entity2_cj})

        return graph



    def _generate_feat_graph(self, args):   ######## 构建同质图，相似性图
        # drug feature graph
        drug_sim = self.drug_sim_features
        drug_num_neighbor = self.num_neighbor
        if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
            drug_num_neighbor = drug_sim.shape[0]

        drug_neighbor = np.argpartition(-drug_sim, kth=drug_num_neighbor, axis=1)[:, :drug_num_neighbor]
        dr_row_index = np.arange(drug_neighbor.shape[0]).repeat(drug_neighbor.shape[1])
        dr_col_index = drug_neighbor.reshape(-1)
        drug_edge_index = np.array([dr_row_index, dr_col_index]).astype(int).T

        drug_edges = np.array(list(drug_edge_index), dtype=np.int32).reshape(drug_edge_index.shape)
        drug_adj = sp.coo_matrix((np.ones(drug_edges.shape[0]), (drug_edges[:, 0], drug_edges[:, 1])),
                                 shape=(self.num_drug, self.num_drug),
                                 dtype=np.float32)
        drug_adj = drug_adj + drug_adj.T.multiply(drug_adj.T > drug_adj) - drug_adj.multiply(
            drug_adj.T > drug_adj)

        drug_deg_np = np.array(drug_adj.sum(axis=1)).reshape(-1)  # shape: [num_drug]
        self.drug_deg = torch.from_numpy(drug_deg_np).float()  # torch.Size([num_drug])
        # drug_graph = normalize(drug_adj + sp.eye(drug_adj.shape[0]))
        drug_graph = normalize(drug_adj)
        drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)

        if not os.path.exists(f"./name_data/drug_data/{args.data_name}/drug_sim_adj.csv"):
            coo = drug_adj.tocoo()
            # 用 DataFrame 保存行、列、值信息
            df = pd.DataFrame({
                'row': coo.row,
                'col': coo.col,
                'data': coo.data
            })

            df.to_csv(f"./name_data/drug_data/{args.data_name}/drug_sim_adj.csv", index=False)

        # disease feature graph
        disease_sim = self.disease_sim_features
        disease_num_neighbor = self.num_neighbor
        if disease_num_neighbor > disease_sim.shape[0] or disease_num_neighbor < 0:
            disease_num_neighbor = disease_sim.shape[0]

        disease_neighbor = np.argpartition(-disease_sim, kth=disease_num_neighbor, axis=1)[:, :disease_num_neighbor]
        di_row_index = np.arange(disease_neighbor.shape[0]).repeat(disease_neighbor.shape[1])
        di_col_index = disease_neighbor.reshape(-1)
        disease_edge_index = np.array([di_row_index, di_col_index]).astype(int).T

        disease_edges = np.array(list(disease_edge_index), dtype=np.int32).reshape(disease_edge_index.shape)
        disease_adj = sp.coo_matrix((np.ones(disease_edges.shape[0]), (disease_edges[:, 0], disease_edges[:, 1])),
                                    shape=(self.num_disease, self.num_disease),
                                    dtype=np.float32)
        disease_adj = disease_adj + disease_adj.T.multiply(disease_adj.T > disease_adj) - disease_adj.multiply(
            disease_adj.T > disease_adj)

        # ====== 计算 disease 的“结构度” ======
        disease_deg_np = np.array(disease_adj.sum(axis=1)).reshape(-1)  # shape: [num_disease]
        self.disease_deg = torch.from_numpy(disease_deg_np).float()  # torch.Size([num_disease])

        # disease_graph = normalize(disease_adj + sp.eye(disease_adj.shape[0]))
        disease_graph = normalize(disease_adj)
        disease_graph = sparse_mx_to_torch_sparse_tensor(disease_graph)

        if not os.path.exists(f"./name_data/drug_data/{args.data_name}/dis_sim_adj.csv"):
            coo = disease_adj.tocoo()
            # 用 DataFrame 保存行、列、值信息
            df = pd.DataFrame({
                'row': coo.row,
                'col': coo.col,
                'data': coo.data
            })

            df.to_csv(f"./name_data/drug_data/{args.data_name}/dis_sim_adj.csv", index=False)

        ####################################################################
        #####################    修改3：load PPI adj   #######################
        ####################################################################

        # protein_sim = self.protein_sim_features
        # protein_num_neighbor = self.num_neighbor * 2
        # if protein_num_neighbor > protein_sim.shape[0] or protein_num_neighbor < 0:
        #     protein_num_neighbor = protein_sim.shape[0]
        # topk_idx = np.argpartition(-protein_sim, kth=protein_num_neighbor, axis=1)[:, :protein_num_neighbor]
        # # 获取对应的相似性得分
        # protein_neighbor_scores = np.take_along_axis(protein_sim, topk_idx, axis=1)
        #
        # # 过滤掉相似度 < 0.4 的邻居
        # THRESH = 0.4
        # protein_neighbor = topk_idx.copy()
        # protein_neighbor[protein_neighbor_scores < THRESH] = -1
        # valid_mask = (protein_neighbor != -1)
        # pi_row_index = np.repeat(np.arange(protein_neighbor.shape[0]), protein_num_neighbor)[valid_mask.ravel()]
        # pi_col_index = protein_neighbor[valid_mask].ravel()
        # protein_edge_index = np.array([pi_row_index, pi_col_index]).astype(int).T
        #
        # protein_edges = np.array(list(protein_edge_index), dtype=np.int32).reshape(protein_edge_index.shape)
        # protein_adj = sp.coo_matrix((np.ones(protein_edges.shape[0]), (protein_edges[:, 0], protein_edges[:, 1])),
        #                             shape=(self.num_protein, self.num_protein),
        #                             dtype=np.float32)
        # protein_adj = protein_adj + protein_adj.T.multiply(protein_adj.T > protein_adj) - protein_adj.multiply(
        #     protein_adj.T > protein_adj)
        # ppi_graph = normalize(protein_adj)
        # ppi_graph = sparse_mx_to_torch_sparse_tensor(ppi_graph)

        ppi_adj = sp.csr_matrix(self.ppi_adj.astype(np.float32))

        # ====== 计算 protein 的“结构度” ======
        protein_deg_np = np.array(ppi_adj.sum(axis=1)).reshape(-1)  # shape: [num_protein]
        self.protein_deg = torch.from_numpy(protein_deg_np).float()  # torch.Size([num_protein])

        ppi_graph = normalize(ppi_adj)
        ppi_graph = sparse_mx_to_torch_sparse_tensor(ppi_graph)

        if not os.path.exists(f"./name_data/drug_data/{args.data_name}/pro_sim_adj-{self.ppi_radio}.csv"):
            coo = ppi_adj.tocoo()
            # 用 DataFrame 保存行、列、值信息
            df = pd.DataFrame({
                'row': coo.row,
                'col': coo.col,
                'data': coo.data
            })

            df.to_csv(f"./name_data/drug_data/{args.data_name}/pro_sim_adj-{self.ppi_radio}.csv", index=False)

        if args.drop_type == "ED":
            assert args.dr_ratio > 0.0
            drug_adj_enhanced = drop_edges_undirected_scipy(drug_adj, drop_rate=args.dr_ratio, keep_self_loop=True)
            drug_graph_enhanced = normalize(drug_adj_enhanced)          # 仍使用你的行归一化
            drug_graph_enhanced = sparse_mx_to_torch_sparse_tensor(drug_graph_enhanced)
            disease_adj_enhanced = drop_edges_undirected_scipy(disease_adj, drop_rate=args.dr_ratio, keep_self_loop=True)
            disease_graph_enhanced = normalize(disease_adj_enhanced)
            disease_graph_enhanced = sparse_mx_to_torch_sparse_tensor(disease_graph_enhanced)
            ppi_adj_enhanced = drop_edges_undirected_scipy(self.ppi_adj, drop_rate=args.dr_ratio, keep_self_loop=True)
            ppi_graph_enhanced = normalize(ppi_adj_enhanced)
            ppi_graph_enhanced = sparse_mx_to_torch_sparse_tensor(ppi_graph_enhanced)
            return drug_graph, disease_graph, ppi_graph, drug_graph_enhanced, disease_graph_enhanced, ppi_graph_enhanced
        else:
            return drug_graph, disease_graph, ppi_graph

    @staticmethod
    def _generate_pair_value(rel_info):
        rating_pairs = (np.array([ele for ele in rel_info["drug_id"]],
                                 dtype=np.int64),
                        np.array([ele for ele in rel_info["disease_id"]],
                                 dtype=np.int64))
        rating_values = rel_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        data_dict = dict()
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rel_values:
            ridx = np.where(
                rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('drug', str(rating), 'disease'): (rrow, rcol),
                ('disease', 'rev-%s' % str(rating), 'drug'): (rcol, rrow)
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            drug_ci = []
            drug_cj = []
            disease_ci = []
            disease_cj = []
            for r in self.possible_rel_values:
                r = to_etype_name(r)
                drug_ci.append(graph['rev-%s' % r].in_degrees())
                disease_ci.append(graph[r].in_degrees())
                if self._symm:
                    drug_cj.append(graph[r].out_degrees())
                    disease_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    drug_cj.append(th.zeros((self.num_drug,)))
                    disease_cj.append(th.zeros((self.num_disease,)))

            drug_ci = _calc_norm(sum(drug_ci))
            disease_ci = _calc_norm(sum(disease_ci))
            if self._symm:
                drug_cj = _calc_norm(sum(drug_cj))
                disease_cj = _calc_norm(sum(disease_cj))
            else:
                drug_cj = th.ones(self.num_drug, )
                disease_cj = th.ones(self.num_disease, )
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_drug, self.num_disease), dtype=np.float32)
        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E',
                                     vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                               num_nodes_dict={'drug': self.num_drug, 'disease': self.num_disease})

    @property
    def num_links(self):
        return self.possible_rel_values.size

    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug

    @property
    def num_protein(self):
        return self._num_protein



if __name__ == '__main__':
    DrugDataLoader("Cdataset", device=th.device('cpu'), symm=True)