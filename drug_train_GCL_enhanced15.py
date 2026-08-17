#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/9/23 13:06
# @Author  : Wenjing
# @File    : drug_train_GCL_enhanced15.py
# @Desc    : train main file

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from model_proj_enhanced15 import Net
from evaluate15 import evaluate, evaluate_GCL
from dataset_loader_enhanced15 import DrugDataLoader
from utils_GCL_proj import *
import wandb
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



####################################################
################ 改动：在14-2的基础上
# 把 PD/DP修改成GAT ###########################
####################################################

def train(args, para_name, dataset, graph_data, cv):
    # wandb
    # wandb.init(
    #     project=f"DDA_enhanced_mt-{args.merge_type}_layer-{args.layers}",
    #     name= str(cv) + "-" + para_name.replace(f"merge_type={args.merge_type}-", "").replace(f"layers={args.layers}-", ""),
    #     config={
    #         "learning_rate": {args.train_lr},
    #         "architecture": "Drug_Rep",
    #         "dataset": args.data_name,
    #         "epochs": args.train_max_iter,
    #         "fold": cv
    #     }
    # )

    # torch.autograd.set_detect_anomaly(True)
    args.src_in_units = dataset.drug_feature_shape[1]  # 1075
    args.dst_in_units = dataset.disease_feature_shape[1]   # 1075
    args.fdim_disease = dataset.disease_feature_shape[0]  # 409
    args.fdim_drug = dataset.drug_feature_shape[0]  # 663
    args.fdim_protein = dataset.num_protein   ##### 993

    dis_graph = dataset.disease_graph.to(args.device)    ##### 相似性图SS
    drug_graph = dataset.drug_graph.to(args.device)      ##### 相似性图DD
    pro_graph = dataset.protein_graph.to(args.device)    ##### 相似性图PP
    if args.drop_type == "ED":
        dis_graph_enhanced = dataset.disease_graph_enhanced.to(args.device)   # 同质图ED
        drug_graph_enhanced = dataset.drug_graph_enhanced.to(args.device)     # 同质图ED
        pro_graph_enhanced = dataset.protein_graph_enhanced.to(args.device)   # 同质图E
    dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(args.device)   ##### 相似性SS矩阵，相似性分数
    drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(args.device)     ##### 相似性DD矩阵，相似性分数
    pro_sim_feat = th.FloatTensor(dataset.protein_sim_features).to(args.device)   ##### 相似性PP矩阵，相似性分数
    pro_embedding = dataset.pro_embedding.to(args.device)    # 从scGPT预训练好的pro_embedding
    args.rating_vals = dataset.possible_rel_values

    # build the model
    model = Net(args=args, protein_scgpt_feat=pro_embedding)
    model = model.to(args.device)
    rel_loss = nn.BCEWithLogitsLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.train_lr)
    print("Loading network finished ...\n")

    # print(model)

    # prepare the logger
    # test_loss_logger = MetricLogger(['iter', 'loss', 'auroc', 'aupr'], ['%d', '%.4f', '%.4f', '%.4f'],
                                    # os.path.join(args.save_dir, 'test_metric%d.csv' % args.save_id))

    # prepare training data
    train_gt_ratings = graph_data['train'][2].to(args.device)
    train_enc_graph = graph_data['train'][0].int().to(args.device)    # 训练构建的DS异质图，
    train_dec_graph = graph_data['train'][1].int().to(args.device)    # 用于解码+loss训练的数据
    drug_feat, dis_feat = dataset.drug_feature, dataset.disease_feature   # 异质图的三列特征
    # pro_feat = dataset.protein_feature   # 不要protein了
    drug_pro_graph = dataset.drug_pro_graph.to(args.device)   # 异质图
    pro_dis_graph = dataset.pro_dis_graph.to(args.device)   # 异质图

    print("Start training ...")

    start = time.perf_counter()
    best_iter, best_auroc, best_aupr = 0, 0, 0

    drug_evi_feat, dis_evi_feat, pro_feat = dataset.drug_evi_feat, dataset.dis_evi_feat, dataset.pro_feat

    for iter_idx in range(1, args.train_max_iter):
        model.train()
        Two_Stage = False

        pred_ratings, drug_out, drug_evidence_out, dis_out, dis_evidence_out, pro_sim_out, pro_e_dout, pro_e_sout, _ = \
            model(train_enc_graph, train_dec_graph,
                  drug_graph, drug_sim_feat, drug_feat,
                  dis_graph, dis_sim_feat, dis_feat,
                  pro_graph, pro_sim_feat,
                  drug_evi_feat, dis_evi_feat, pro_feat,
                  drug_pro_graph, pro_dis_graph,
                  Two_Stage)

        pred_ratings = pred_ratings.squeeze(-1)

        drug_out_cl_proj = model.cl_drug_proj(drug_out)
        drug_evidence_out_cl_proj = model.cl_drug_proj(drug_evidence_out)
        dis_out_cl_proj = model.cl_dis_proj(dis_out)
        dis_evidence_out_cl_proj = model.cl_dis_proj(dis_evidence_out)

        loss_drug = LOSS(args, drug_out_cl_proj, drug_evidence_out_cl_proj, batch_size=0, flag=0)
        loss_dis = LOSS(args, dis_out_cl_proj, dis_evidence_out_cl_proj, batch_size=0, flag=1)

        if args.loss_protein_add:
            # 为CL的protein 的投影层
            pro_e_dout_proj = model.cl_protein_proj(pro_e_dout)
            pro_e_sout_proj = model.cl_protein_proj(pro_e_sout)
            pro_sim_out_proj = model.cl_protein_proj(pro_sim_out)
            protein_scgpt_proj = model.cl_pro_scgpt_proj(pro_embedding)

            loss_pro = LOSS_multi_2(args, pro_sim_out_proj, pro_e_dout_proj, pro_e_sout_proj, protein_scgpt_proj, batch_size=0, flag=2)

            loss_bce = rel_loss(pred_ratings, train_gt_ratings)
            ########### 加入margin loss
            margin = args.margin  # 先固定，不暴露成超参
            pos_mask = (train_gt_ratings == 1)
            neg_mask = (train_gt_ratings == 0)

            pos_scores = pred_ratings[pos_mask]
            neg_scores = pred_ratings[neg_mask]

            if len(pos_scores) > 0 and len(neg_scores) > 0:
                # 给每个正例随机配一个负例
                neg_idx = torch.randint(0, len(neg_scores), (len(pos_scores),),
                                        device=args.device)
                neg_sample = neg_scores[neg_idx]
                margin_loss = F.relu(neg_sample - pos_scores + margin).mean()
            else:
                margin_loss = torch.tensor(0., device=args.device)

            # 3) 总 loss：多了一个 very small 的 lambda_auc
            loss = args.beta * (loss_drug + loss_dis + loss_pro) / 3 + loss_bce + args.lambda_margin * margin_loss
        else:
            loss_bce = rel_loss(pred_ratings, train_gt_ratings)
            loss = args.beta * (loss_drug + loss_dis) / 2 + loss_bce
            loss_pro = 0.0
            # pro_reg = 0.0

        # loss = ( loss_drug +  loss_dis )/2
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()

        auroc, aupr, y_true, y_score = evaluate_GCL(args, model, graph_data,
                                                drug_graph, drug_feat, drug_sim_feat,
                                                dis_graph, dis_feat, dis_sim_feat,
                                                pro_graph, pro_sim_feat,
                                                drug_evi_feat, dis_evi_feat, pro_feat,
                                                drug_pro_graph, pro_dis_graph)
        # test_loss_logger.log(iter=iter_idx, loss=loss.item(), auroc=auroc, aupr=aupr)
        logging_str = "Iter={}, loss={:.4f}==loss_bce: {:.4f} + loss_drug: {:.4f} + loss_dis: {:.4f} + loss_pro: {:.4f} AUROC={:.4f}, AUPR={:.4f}".format(
            iter_idx, loss.item(), loss_bce.item(), loss_drug.item(), loss_dis.item(), loss_pro.item(), auroc, aupr)
        # wandb.log(
        #     {
        #         "loss": loss.item(),
        #         "loss_bce": loss_bce.item(),
        #         "loss_drug": loss_drug.item(),
        #         "loss_dis": loss_dis.item(),
        #         "loss_pro": loss_pro.item(),
        #         "AUPR": aupr,
        #     }
        # )
        if auroc > best_auroc:
            best_iter, best_auroc, best_aupr, true, score = iter_idx, auroc, aupr, y_true, y_score
            path = os.path.join(args.model_path, para_name + '-' + str(cv+1)+'.pkl')
            th.save(model, path)
        if iter_idx % args.train_valid_interval == 0:
            print("test-logging_str", logging_str)

    end = time.perf_counter()

    print("running time", time.strftime("%H:%M:%S", time.gmtime(round(end - start))))
    print("Bset_Iter={}, Best_AUROC={:.4f}, Best_AUPR={:.4f}".format(best_iter, best_auroc, best_aupr))
    print(f"model save to {path}")
    # test_loss_logger.close()
    return best_auroc, best_aupr

def params_to_string(args, keys=None):
    """
    Produce a compact, stable string representing hyperparameters.
    If keys is None, use a default ordered list of relevant args.
    """
    if keys is None:
        # keys = [
        #     "train_lr",
        #     "lambda_margin",
        #     # "inner_concat_type", "intra_concat_type",
        #     # "reg",
        #     "E_layers", "ppi_radio",
        #     "beta", "tau",
        #     # "gcn_out_units", "gcn_agg_units", "nhid1", "nhid2", "layers",
        #     "dropout", "emb_dropout",
        #     "num_neighbor",
        #     "merge_type",
        #     "loss_protein_add",
        # ]
        keys = [
            "train_lr",
            "gcn_out_units", "gcn_agg_units",
            # "nhid1", "nhid2",
            "dropout", "emb_dropout",
            "lambda_margin",
            # # "inner_concat_type", "intra_concat_type",
            # # "reg",
            "layers",
            "E_layers",
            "ppi_radio",
            "beta",
            "tau",
            #
            "num_neighbor",
            "train_max_iter"
            # "merge_type",
            # "loss_protein_add",
        ]
    parts = []
    for k in keys:
        v = getattr(args, k, None)
        parts.append(f"{k}={v}")
    return "-".join(parts)


'''
改动： 添加GCL的node dropout ND, edge dropout ED, ori OP, 以及他们的ratio
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XMMBNet')
    parser.add_argument('--seed', default=125, type=int)   # 1024， 1234
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--data_name', default='Adataset', type=str)
    parser.add_argument('--model_activation', type=str, default="tanh")
    parser.add_argument('--dropout', type=float, default=0.25)    # 0.1， 0.2， 0.25， 0.3， 0.35
    parser.add_argument('--gcn_agg_units', type=int, default=840)   # 420，600, 900, 1620, 1800
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=75)  # 16, 32, 64, 75, 128
    parser.add_argument('--train_max_iter', type=int, default=5000)
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_valid_interval', type=int, default=100)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--beta', type=float, default=0.1)    # 0.001, 0.01, 0.1, 1, 10, 100, autoloss
    parser.add_argument('--num_neighbor', type=int, default=20)  # 1, 4, 6, 8, 10, 12, 14, 16, 18, 20
    parser.add_argument('--nhid1', type=int, default=500)   # 同质图GCM的 hidden dim1 500, 600, 700, 800, 900
    parser.add_argument('--nhid2', type=int, default=75)   # 同质图GCM的 output dim2 16, 32, 64, 75, 128
    parser.add_argument('--train_lr', type=float, default=0.03)  # 1e-2, 1e-3, 1e-1, 1e-4, 1e-5 / 确定其他参数后，在该数量级进行调参
    parser.add_argument('--layers', type=int, default=1)   # 1,2,3,4,5
    parser.add_argument('--E_layers', type=int, default=1)   # evidence layer 调参 # 1,2,3,4,5
    parser.add_argument('--stack_layers', type=int, default=3)  # 调参 # 1,2,3,4,5
    parser.add_argument('--tau', type=float, default=0.1)   # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
    parser.add_argument('--intra', type=float, default=0.2)
    parser.add_argument('--inter', type=float, default=0.5)
    parser.add_argument('--share_param', default=True, action='store_true')
    ### projection 是否需要加入到net中进行训练
    parser.add_argument('--projection_train', default=True, type=bool)
    parser.add_argument('--save_name', type=str, default="lr0.01")
    ### GCL 新增参数
    parser.add_argument('--drop_type', type=str, default="OP")   # OP, ED, ND
    parser.add_argument('--dr_ratio', type=float, default=0.0)  # OP, ED, ND
    parser.add_argument('--inner_concat_type', type=str, default="gate")   # 视角内实体的 拼接方式, stack-MLP, mean-加和求平均, attention, gate
    parser.add_argument('--intra_concat_type', type=str, default="xattn")  # 跨视角 实体的 拼接方式, stack-MLP, mean-加和求平均, attention, gate
    parser.add_argument('--merge_type', type=str, default="fusion")  # 不同view的融合方式, stack-MLP， 不加操作，全部丢到MLP里面去，
                                                                                  # fusion: 先融合Evidence view, 再和direct view拼起来放到MLP里面算分数
                                                                                  # fusion_CV:  先融合Evidence view, 然后两个view分别进MLP算分数，加和
    parser.add_argument('--use_pro', default=True, type=bool)
    parser.add_argument('--emb_dropout', type=float, default=0.1)  # 0.1， 0.2， 0.25， 0.3， 0.35
    parser.add_argument('--ppi_radio', type=float, default=0.4)  # 0.4, 0.6, 0.7
    parser.add_argument('--loss_protein', default=False, type=bool)
    parser.add_argument('--loss_protein_add', default=True, type=bool)
    parser.add_argument('--lambda_margin', type=float, default=0.02)  # OP, ED, ND
    parser.add_argument('--margin', type=float, default=0.05)

    args = parser.parse_args()
    print(args)
    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    aucs, auprs = [], []
    para_name = params_to_string(args)
    para_name = f"15-{para_name}"

    all_result_path = os.path.join("result", args.data_name)

    if not os.path.isdir(all_result_path):
        os.makedirs(all_result_path)
    file_path_times = all_result_path + f"/{args.save_name}_all.xlsx"  # 0,1
    if not os.path.exists(file_path_times):
        # 创建一个DataFrame
        columns = ["Parameter", "AUC", "AUPR", "AUC_list", "AUPR_list"]
        df_times = pd.DataFrame(columns=columns)
        df_times.to_excel(file_path_times, index=False)
    # 读取现有的Excel文件
    df_times = pd.read_excel(file_path_times)


    for times in range(0, 1):
        print("++++++++++++++++++times", str(times), "++++++++++++++++++++++")
        args.save_dir = args.data_name + "_" + ''.join(str(times) + 'time')
        args.save_dir = os.path.join("result", args.save_dir)

        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)

        file_path = args.save_dir + f"/{args.save_name}.xlsx"
        if not os.path.exists(file_path):
            # 创建一个DataFrame
            columns = ["Parameter", "AUC", "AUPR", "AUC_list", "AUPR_list"]
            df = pd.DataFrame(columns=columns)
            df.to_excel(file_path, index=False)
            # 读取现有的Excel文件
        df = pd.read_excel(file_path)

        args.model_path = os.path.join("weight", args.data_name + "_" + ''.join(str(times) + 'time'))
        if not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)

        dataset = DrugDataLoader(args, args.data_name, args.device,
                                 symm=args.gcn_agg_norm_symm,
                                 k=args.num_neighbor)

        print("Loading dataset finished ...\n")

        auc_list, aupr_list = [], []
        for cv in range(0, 10):
            args.save_id = cv + 1
            print("===============" + str(cv + 1) + "=================")
            graph_data = dataset.data_cv[cv]
            auc, aupr = train(args, para_name, dataset, graph_data, cv)
            auc_list.append(round(auc, 4))
            aupr_list.append(round(aupr, 4))

        print("Mean_AUROC{:4f}".format(np.mean(auc_list)), "Mean_AURP{:4f}".format(np.mean(aupr_list)))
        print("auroc_list", auc_list)
        print("aupr_list", aupr_list)
        # 写入文件
        new_df = pd.DataFrame(
            [[para_name, np.mean(auc_list), np.mean(aupr_list), auc_list, aupr_list]],
             columns=df.columns
        )
        df = pd.concat([df, new_df], ignore_index=True)
        # 写入Excel文件
        df.to_excel(file_path, index=False)
        aucs += auc_list
        auprs += aupr_list
    print("mean times auc{:4f} ".format(np.mean(aucs)),
          "mean times aupr{:4f} ".format(np.mean(auprs)))
    print("aucs", aucs)
    print("auprs", auprs)
    # 写入文件
    new_df_times = pd.DataFrame(
        [[para_name, np.mean(aucs), np.mean(auprs), aucs, auprs]],
        columns=df_times.columns
    )
    df_times = pd.concat([df_times, new_df_times], ignore_index=True)
    # 写入Excel文件
    df_times.to_excel(file_path_times, index=False)

