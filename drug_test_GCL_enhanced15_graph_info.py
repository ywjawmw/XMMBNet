import os
import time
import argparse
import numpy as np
import pandas as pd
import torch as th

from model_proj_enhanced15 import Net
from evaluate15_test import evaluate_GCL
from dataset_loader_enhanced15 import DrugDataLoader
from utils_GCL_proj import *
from collections import defaultdict
import wandb
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



####################################################
################ 改动：在14-2的基础上
# 把 PD/DP的GCMC修改成GAT ###########################
####################################################

def train(args, para_name, dataset, graph_data, cv, case_list_id):
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
    args.model_path = os.path.join("weight", args.data_name + "_" + ''.join(str(times) + 'time'))
    path = os.path.join(args.model_path, para_name + '-' + str(cv + 1) + '.pkl')
    model = torch.load(path, map_location=lambda storage, loc: storage)
    model.protein_scgpt_feat = model.protein_scgpt_feat.to(args.device)
    model = model.to(args.device)
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

    print("Start testing ...")

    start = time.perf_counter()

    drug_evi_feat, dis_evi_feat, pro_feat = dataset.drug_evi_feat, dataset.dis_evi_feat, dataset.pro_feat

    auroc, aupr, y_true, y_score, topk_drug_id, topk_score, exp_dis, explain_records_drug = evaluate_GCL(args, model, graph_data,
                                                drug_graph, drug_feat, drug_sim_feat,
                                                dis_graph, dis_feat, dis_sim_feat,
                                                pro_graph, pro_sim_feat,
                                                drug_evi_feat, dis_evi_feat, pro_feat,
                                                drug_pro_graph, pro_dis_graph, case_list_id)
    # test_loss_logger.log(iter=iter_idx, loss=loss.item(), auroc=auroc, aupr=aupr)
    logging_str = "AUROC={:.4f}, AUPR={:.4f}".format(auroc, aupr)
    print(logging_str)

    end = time.perf_counter()

    print("running time", time.strftime("%H:%M:%S", time.gmtime(round(end - start))))
    # test_loss_logger.close()
    return auroc, aupr, topk_drug_id, topk_score, exp_dis, explain_records_drug

def params_to_string(args, keys=None):
    """
    Produce a compact, stable string representing hyperparameters.
    If keys is None, use a default ordered list of relevant args.
    """
    if keys is None:
        # ### one best
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
        ### two best
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


def fold_topk_to_df(topk_drug_id, topk_score, drug_id2bank, bank2name, exp_dis, explain_records_drug):
    rows = []
    rows.append([
        0,
        "OMIM",
        "disease",
        0.0,
        exp_dis["sim_score"],
        exp_dis["pro_score"],
        exp_dis["dir_score"],
        exp_dis["evi_score"]
    ])
    for rank, (did, score) in enumerate(zip(topk_drug_id, topk_score), start=1):
        did = did.item()
        exp_drug = explain_records_drug.get(did)
        drugbank_id = drug_id2bank.get(did, "NA")
        drug_name = bank2name.get(drugbank_id, "NA")
        rows.append([
            rank,
            drugbank_id,
            drug_name,
            float(score),
            exp_drug["sim_score"],
            exp_drug["pro_score"],
            exp_drug["dir_score"],
            exp_drug["evi_score"]
        ])
    return pd.DataFrame(
        rows,
        columns=["rank", "drugbank_id", "drug_name", "score", "sim_score", "pro_score", "dir_score", "evi_score"]
    )


def load_graph_name_maps(data_name):
    data_dir = f"./name_data/drug_data/{data_name}"

    disease_map = pd.read_csv(os.path.join(data_dir, "disease_name.csv"))
    disease_id2name = dict(zip(disease_map["disease_id"], disease_map["disease_name"]))

    protein_path = os.path.join(data_dir, "protein_name_enhanced.csv")
    if not os.path.exists(protein_path):
        protein_path = os.path.join(data_dir, "protein_name.csv")
    protein_map = pd.read_csv(protein_path)
    protein_id2name = dict(zip(protein_map["protein_id"], protein_map["protein_name"]))
    if "hgnc_symbol" in protein_map.columns:
        protein_id2symbol = dict(zip(protein_map["protein_id"], protein_map["hgnc_symbol"]))
    else:
        protein_id2symbol = protein_id2name.copy()

    return disease_id2name, protein_id2name, protein_id2symbol


def resolve_graph_info_disease_id(data_name, disease_id, disease_name):
    if disease_name is None:
        return disease_id

    disease_map = pd.read_csv(f"./name_data/drug_data/{data_name}/disease_name.csv")
    matched = disease_map[disease_map["disease_name"].astype(str) == str(disease_name)]
    if matched.empty:
        raise ValueError(f"Cannot find disease_name={disease_name} in disease_name.csv")
    return int(matched.iloc[0]["disease_id"])


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _top_positive_indices(values, max_items=None, skip_self=None):
    idx = np.flatnonzero(values > 0).tolist()
    if skip_self is not None:
        idx = [i for i in idx if i != skip_self]
    idx = sorted(idx, key=lambda i: values[i], reverse=True)
    if max_items is not None:
        idx = idx[:max_items]
    return idx


def _protein_display(protein_id, protein_id2name, protein_id2symbol):
    protein_name = protein_id2name.get(int(protein_id), f"protein_id:{protein_id}")
    protein_symbol = protein_id2symbol.get(int(protein_id), protein_name)
    return protein_name, protein_symbol, f"{protein_symbol}({protein_name})"


def _ppi_adjacency(ppi_adj):
    matrix = np.asarray(ppi_adj)
    rows, cols = np.nonzero(matrix > 0)
    adjacency = defaultdict(set)
    for row, col in zip(rows, cols):
        row, col = int(row), int(col)
        if row == col:
            continue
        adjacency[row].add(col)
        adjacency[col].add(row)
    return adjacency


def _shortest_path_within_hops(adjacency, start, targets, max_hops=3):
    targets = set(int(x) for x in targets)
    if start in targets:
        return [int(start)]

    queue = [(int(start), [int(start)])]
    visited = {int(start)}
    while queue:
        node, path = queue.pop(0)
        if len(path) - 1 >= max_hops:
            continue
        for neighbor in sorted(adjacency.get(node, [])):
            if neighbor in visited:
                continue
            next_path = path + [neighbor]
            if neighbor in targets:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path))
    return None


def _paths_from_source_within_hops(adjacency, start, max_hops=3):
    paths = {int(start): [int(start)]}
    queue = [(int(start), [int(start)])]
    while queue:
        node, path = queue.pop(0)
        if len(path) - 1 >= max_hops:
            continue
        for neighbor in sorted(adjacency.get(node, [])):
            if neighbor in paths:
                continue
            next_path = path + [neighbor]
            paths[neighbor] = next_path
            queue.append((neighbor, next_path))
    return paths


def _format_protein_path(path, protein_id2name, protein_id2symbol):
    return " -> ".join(
        _protein_display(pid, protein_id2name, protein_id2symbol)[2]
        for pid in path
    )


def resolve_selected_top_drug(top_summary_df, args, drug_id2bank):
    selected = top_summary_df[top_summary_df["summary_rank"] == args.protein_path_top_id]
    if selected.empty:
        raise ValueError(
            "Selected top ID is not in the exported Summary top list. "
            "For example, --protein_path_top_id 1 means the first drug in Summary_MeanRank. "
            "Use --graph_info_topn to include a larger rank if needed."
        )
    return selected.iloc[0]


def build_single_drug_disease_ppi_info(
    args,
    dataset,
    top_summary_df,
    drug_id2bank,
    bank2name,
):
    disease_id2name, protein_id2name, protein_id2symbol = load_graph_name_maps(args.data_name)
    selected_drug = resolve_selected_top_drug(top_summary_df, args, drug_id2bank)

    drug_id = int(selected_drug["drug_id"])
    drugbank_id = selected_drug["drugbank_id"]
    drug_name = selected_drug["drug_name"]
    summary_rank = int(selected_drug["summary_rank"])
    disease_id = int(args.graph_info_disease_id)
    disease_name = disease_id2name.get(disease_id, f"disease_id:{disease_id}")

    drug_proteins = set(np.flatnonzero(dataset.drug_pro_matrix[drug_id] > 0).astype(int).tolist())
    disease_proteins = set(np.flatnonzero(dataset.pro_dis_matrix[:, disease_id] > 0).astype(int).tolist())
    overlap_proteins = sorted(drug_proteins & disease_proteins)
    ppi_adj = _ppi_adjacency(dataset.ppi_adj)

    status = {
        "selected_summary_rank": summary_rank,
        "drug_id": drug_id,
        "drugbank_id": drugbank_id,
        "drug_name": drug_name,
        "disease_id": disease_id,
        "disease_name": disease_name,
        "drug_protein_count": len(drug_proteins),
        "disease_protein_count": len(disease_proteins),
        "overlap_protein_count": len(overlap_proteins),
        "ppi_max_hops": 3,
        "ppi_radio": args.ppi_radio,
        "result_type": None,
        "message": None,
    }

    if len(drug_proteins) == 0 or len(disease_proteins) == 0:
        status["result_type"] = "no_related_protein"
        status["message"] = "无相关protein：drug-protein或protein-disease图中没有可分析的protein。"
        return pd.DataFrame([status]), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if overlap_proteins:
        status["result_type"] = "overlap_protein"
        status["message"] = "drug-protein和protein-disease图存在重合protein，已输出重合protein及其PPI三阶邻居。"
        overlap_rows = []
        neighbor_rows = []
        summary_rows = []
        for overlap_id in overlap_proteins:
            protein_name, protein_symbol, protein_display = _protein_display(overlap_id, protein_id2name, protein_id2symbol)
            overlap_rows.append({
                **status,
                "overlap_protein_id": overlap_id,
                "overlap_protein_name": protein_name,
                "overlap_protein_symbol": protein_symbol,
                "overlap_protein_display": protein_display,
            })
            summary_rows.append({
                "drug_protein": protein_display,
                "overlap_protein": protein_display,
                "reachable_disease_protein_list": "",
                "drug_protein_id": overlap_id,
                "overlap_protein_id": overlap_id,
                "reachable_disease_protein_ids": "",
            })

            reachable = _paths_from_source_within_hops(ppi_adj, overlap_id, max_hops=3)
            for neighbor_id, path in sorted(reachable.items(), key=lambda item: (len(item[1]) - 1, item[0])):
                if neighbor_id == overlap_id:
                    continue
                neighbor_name, neighbor_symbol, neighbor_display = _protein_display(
                    neighbor_id, protein_id2name, protein_id2symbol
                )
                neighbor_rows.append({
                    "selected_summary_rank": summary_rank,
                    "drug_id": drug_id,
                    "drugbank_id": drugbank_id,
                    "drug_name": drug_name,
                    "disease_id": disease_id,
                    "disease_name": disease_name,
                    "overlap_protein_id": overlap_id,
                    "overlap_protein_display": protein_display,
                    "neighbor_protein_id": neighbor_id,
                    "neighbor_protein_name": neighbor_name,
                    "neighbor_protein_symbol": neighbor_symbol,
                    "neighbor_protein_display": neighbor_display,
                    "hop_distance": len(path) - 1,
                    "ppi_path_ids": " -> ".join(map(str, path)),
                    "ppi_path": _format_protein_path(path, protein_id2name, protein_id2symbol),
                })
        return pd.DataFrame(overlap_rows), pd.DataFrame(neighbor_rows), pd.DataFrame(), pd.DataFrame(summary_rows)

    path_rows = []
    for drug_protein_id in sorted(drug_proteins):
        reachable = _paths_from_source_within_hops(ppi_adj, drug_protein_id, max_hops=3)
        for disease_protein_id in sorted(disease_proteins):
            path = reachable.get(disease_protein_id)
            if path is None:
                continue
            dp_name, dp_symbol, dp_display = _protein_display(drug_protein_id, protein_id2name, protein_id2symbol)
            disease_p_name, disease_p_symbol, disease_p_display = _protein_display(
                disease_protein_id, protein_id2name, protein_id2symbol
            )
            path_rows.append({
                "selected_summary_rank": summary_rank,
                "drug_id": drug_id,
                "drugbank_id": drugbank_id,
                "drug_name": drug_name,
                "disease_id": disease_id,
                "disease_name": disease_name,
                "drug_side_protein_id": drug_protein_id,
                "drug_side_protein_name": dp_name,
                "drug_side_protein_symbol": dp_symbol,
                "drug_side_protein_display": dp_display,
                "disease_side_protein_id": disease_protein_id,
                "disease_side_protein_name": disease_p_name,
                "disease_side_protein_symbol": disease_p_symbol,
                "disease_side_protein_display": disease_p_display,
                "hop_distance": len(path) - 1,
                "ppi_path_ids": " -> ".join(map(str, path)),
                "ppi_path": _format_protein_path(path, protein_id2name, protein_id2symbol),
            })

    if path_rows:
        status["result_type"] = "ppi_path_within_3_hops"
        status["message"] = "drug-protein和protein-disease图没有重合protein，但在PPI三阶内可达，已输出路径。"
        path_df = pd.DataFrame(path_rows)
        summary_rows = []
        for drug_protein_id, group in path_df.groupby("drug_side_protein_id"):
            drug_display = group.iloc[0]["drug_side_protein_display"]
            disease_protein_ids = sorted(group["disease_side_protein_id"].astype(int).unique().tolist())
            disease_protein_list = []
            for disease_protein_id in disease_protein_ids:
                _, _, disease_display = _protein_display(disease_protein_id, protein_id2name, protein_id2symbol)
                disease_protein_list.append(disease_display)
            summary_rows.append({
                "drug_protein": drug_display,
                "overlap_protein": "",
                "reachable_disease_protein_list": "; ".join(disease_protein_list),
                "drug_protein_id": int(drug_protein_id),
                "overlap_protein_id": "",
                "reachable_disease_protein_ids": "; ".join(map(str, disease_protein_ids)),
            })
        return pd.DataFrame([status]), pd.DataFrame(), path_df, pd.DataFrame(summary_rows)

    status["result_type"] = "no_related_protein"
    status["message"] = "无相关protein：drug-protein和protein-disease无重合protein，且PPI三阶内不可达。"
    return pd.DataFrame([status]), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def build_top_drug_graph_info(
    args,
    dataset,
    summary_df,
    fold_dfs,
    drug_id2bank,
    bank2name,
    topn=10,
    fold=1,
):
    if fold < 1 or fold > len(fold_dfs):
        raise ValueError(f"graph_info_fold must be in [1, {len(fold_dfs)}], got {fold}")

    disease_id2name, protein_id2name, protein_id2symbol = load_graph_name_maps(args.data_name)
    bank2drug_id = {bank: did for did, bank in drug_id2bank.items()}
    fold_df = fold_dfs[fold - 1]
    train_df = dataset.cv_data_dict[fold - 1][0]

    detail_rows = []
    top_summary = summary_df.head(topn).copy()
    top_summary.insert(0, "summary_rank", range(1, len(top_summary) + 1))
    top_summary["drug_id"] = top_summary["drugbank_id"].map(bank2drug_id)
    top_summary["selected_fold"] = fold

    for _, summary_row in top_summary.iterrows():
        drug_id = int(summary_row["drug_id"])
        drugbank_id = summary_row["drugbank_id"]
        drug_name = summary_row["drug_name"]
        summary_rank = int(summary_row["summary_rank"])

        fold_match = fold_df[fold_df["drugbank_id"] == drugbank_id]
        if fold_match.empty:
            fold_rank = None
            fold_score = None
            score_values = {"sim_score": None, "pro_score": None, "dir_score": None, "evi_score": None}
        else:
            fold_row = fold_match.iloc[0]
            fold_rank = int(fold_row["rank"])
            fold_score = _safe_float(fold_row["score"])
            score_values = {
                "sim_score": _safe_float(fold_row["sim_score"]),
                "pro_score": _safe_float(fold_row["pro_score"]),
                "dir_score": _safe_float(fold_row["dir_score"]),
                "evi_score": _safe_float(fold_row["evi_score"]),
            }

        def add_row(score_name, graph_name, edge_type, neighbor_id, neighbor_key, neighbor_name, edge_weight, sentence):
            detail_rows.append({
                "summary_rank": summary_rank,
                "drug_id": drug_id,
                "drugbank_id": drugbank_id,
                "drug_name": drug_name,
                "fold": fold,
                "fold_rank": fold_rank,
                "fold_score": fold_score,
                "score_name": score_name,
                "score_value": score_values.get(score_name),
                "graph": graph_name,
                "edge_type": edge_type,
                "neighbor_id": int(neighbor_id),
                "neighbor_key": neighbor_key,
                "neighbor_name": neighbor_name,
                "edge_weight": float(edge_weight),
                "sentence": sentence,
            })

        # sim_score: drug-drug similarity graph.
        drug_sim = dataset.drug_sim_features[drug_id]
        for nb_drug_id in _top_positive_indices(drug_sim, max_items=args.num_neighbor, skip_self=drug_id):
            nb_bank = drug_id2bank.get(nb_drug_id, f"drug_id:{nb_drug_id}")
            nb_name = bank2name.get(nb_bank, nb_bank)
            add_row(
                "sim_score",
                "drug_sim",
                "drug-drug similarity",
                nb_drug_id,
                nb_bank,
                nb_name,
                drug_sim[nb_drug_id],
                f"{drugbank_id}({drug_name}) 在drug-drug相似图上关联的相似药物是 {nb_bank}({nb_name})。",
            )

        # pro_score/evi_score: drug-protein evidence graph.
        drug_pro = dataset.drug_pro_matrix[drug_id]
        for score_name in ["pro_score", "evi_score"]:
            for protein_id in _top_positive_indices(drug_pro):
                protein_name = protein_id2name.get(protein_id, f"protein_id:{protein_id}")
                protein_symbol = protein_id2symbol.get(protein_id, protein_name)
                add_row(
                    score_name,
                    "drug_pro",
                    "drug-protein evidence",
                    protein_id,
                    protein_name,
                    protein_symbol,
                    drug_pro[protein_id],
                    f"{drugbank_id}({drug_name}) 在drug-pro图上关联的蛋白是 {protein_symbol}({protein_name})。",
                )

        # dir_score: positive drug-disease edges in this fold's training graph.
        direct_rows = train_df[(train_df["drug_id"] == drug_id) & (train_df["values"] == 1)]
        for _, rel_row in direct_rows.iterrows():
            disease_id = int(rel_row["disease_id"])
            disease_name = disease_id2name.get(disease_id, f"disease_id:{disease_id}")
            add_row(
                "dir_score",
                "fold_train_drug_dis",
                "drug-disease direct",
                disease_id,
                disease_name,
                disease_name,
                1.0,
                f"{drugbank_id}({drug_name}) 在指定fold的drug-dis训练图上关联的疾病/症状是 {disease_name}。",
            )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        summary_graph_df = detail_df
    else:
        summary_graph_df = (
            detail_df
            .groupby(
                [
                    "summary_rank", "drug_id", "drugbank_id", "drug_name", "fold",
                    "fold_rank", "fold_score", "score_name", "score_value", "graph", "edge_type"
                ],
                dropna=False
            )
            .agg(
                neighbor_count=("neighbor_id", "count"),
                neighbors=("neighbor_name", lambda x: "; ".join(map(str, x))),
                neighbor_keys=("neighbor_key", lambda x: "; ".join(map(str, x))),
            )
            .reset_index()
        )

    return top_summary, summary_graph_df, detail_df


def build_disease_graph_info(
    args,
    dataset,
    fold_dfs,
    drug_id2bank,
    bank2name,
    disease_id,
    fold=1,
):
    if fold < 1 or fold > len(fold_dfs):
        raise ValueError(f"graph_info_fold must be in [1, {len(fold_dfs)}], got {fold}")
    if disease_id < 0 or disease_id >= dataset.num_disease:
        raise ValueError(f"graph_info_disease_id must be in [0, {dataset.num_disease - 1}], got {disease_id}")

    disease_id2name, protein_id2name, protein_id2symbol = load_graph_name_maps(args.data_name)
    fold_df = fold_dfs[fold - 1]
    train_df = dataset.cv_data_dict[fold - 1][0]
    disease_name = disease_id2name.get(disease_id, f"disease_id:{disease_id}")

    disease_fold_rows = fold_df[fold_df["rank"] == 0]
    if disease_fold_rows.empty:
        score_values = {"sim_score": None, "pro_score": None, "dir_score": None, "evi_score": None}
    else:
        disease_fold_row = disease_fold_rows.iloc[0]
        score_values = {
            "sim_score": _safe_float(disease_fold_row["sim_score"]),
            "pro_score": _safe_float(disease_fold_row["pro_score"]),
            "dir_score": _safe_float(disease_fold_row["dir_score"]),
            "evi_score": _safe_float(disease_fold_row["evi_score"]),
        }

    detail_rows = []

    def add_row(score_name, graph_name, edge_type, neighbor_id, neighbor_key, neighbor_name, edge_weight, sentence):
        detail_rows.append({
            "disease_id": disease_id,
            "disease_name": disease_name,
            "fold": fold,
            "score_name": score_name,
            "score_value": score_values.get(score_name),
            "graph": graph_name,
            "edge_type": edge_type,
            "neighbor_id": int(neighbor_id),
            "neighbor_key": neighbor_key,
            "neighbor_name": neighbor_name,
            "edge_weight": float(edge_weight),
            "sentence": sentence,
        })

    # sim_score: disease-disease similarity graph.
    disease_sim = dataset.disease_sim_features[disease_id]
    for nb_disease_id in _top_positive_indices(disease_sim, max_items=args.num_neighbor, skip_self=disease_id):
        nb_disease_name = disease_id2name.get(nb_disease_id, f"disease_id:{nb_disease_id}")
        add_row(
            "sim_score",
            "disease_sim",
            "disease-disease similarity",
            nb_disease_id,
            nb_disease_name,
            nb_disease_name,
            disease_sim[nb_disease_id],
            f"{disease_name} 在disease-disease相似图上关联的相似疾病/症状是 {nb_disease_name}。",
        )

    # pro_score/evi_score: protein-disease evidence graph.
    pro_dis_col = dataset.pro_dis_matrix[:, disease_id]
    for score_name in ["pro_score", "evi_score"]:
        for protein_id in _top_positive_indices(pro_dis_col):
            protein_name = protein_id2name.get(protein_id, f"protein_id:{protein_id}")
            protein_symbol = protein_id2symbol.get(protein_id, protein_name)
            add_row(
                score_name,
                "pro_dis",
                "protein-disease evidence",
                protein_id,
                protein_name,
                protein_symbol,
                pro_dis_col[protein_id],
                f"{disease_name} 在pro-dis图上关联的蛋白是 {protein_symbol}({protein_name})。",
            )

    # dir_score: positive drug-disease edges in this fold's training graph.
    direct_rows = train_df[(train_df["disease_id"] == disease_id) & (train_df["values"] == 1)]
    for _, rel_row in direct_rows.iterrows():
        drug_id = int(rel_row["drug_id"])
        drugbank_id = drug_id2bank.get(drug_id, f"drug_id:{drug_id}")
        drug_name = bank2name.get(drugbank_id, drugbank_id)
        add_row(
            "dir_score",
            "fold_train_drug_dis",
            "drug-disease direct",
            drug_id,
            drugbank_id,
            drug_name,
            1.0,
            f"{disease_name} 在指定fold的drug-dis训练图上关联的药物是 {drugbank_id}({drug_name})。",
        )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        summary_df = detail_df
    else:
        summary_df = (
            detail_df
            .groupby(
                [
                    "disease_id", "disease_name", "fold", "score_name",
                    "score_value", "graph", "edge_type"
                ],
                dropna=False
            )
            .agg(
                neighbor_count=("neighbor_id", "count"),
                neighbors=("neighbor_name", lambda x: "; ".join(map(str, x))),
                neighbor_keys=("neighbor_key", lambda x: "; ".join(map(str, x))),
            )
            .reset_index()
        )

    disease_info_df = pd.DataFrame([{
        "disease_id": disease_id,
        "disease_name": disease_name,
        "fold": fold,
        "sim_score": score_values["sim_score"],
        "pro_score": score_values["pro_score"],
        "dir_score": score_values["dir_score"],
        "evi_score": score_values["evi_score"],
    }])
    return disease_info_df, summary_df, detail_df


def run_10fold_and_screen(
    args,
    para_name,
    dataset,
    drug_id2bank,
    bank2name,
    topk=20,
    save_path="TopK_drug_screening.xlsx",
    case_list_id = [439]
):
    auc_list, aupr_list = [], []

    # 用于 ensemble
    all_fold_results = []
    fold_dfs = []

    writer = pd.ExcelWriter(save_path, engine="openpyxl")

    for cv in range(10):
        args.save_id = cv + 1
        print(f"=============== Fold {cv+1} ===============")

        graph_data = dataset.data_cv[cv]

        auc, aupr, topk_drug_id, topk_score, exp_dis, explain_records_drug = train(
            args, para_name, dataset, graph_data, cv, case_list_id
        )

        auc_list.append(round(auc, 4))
        aupr_list.append(round(aupr, 4))

        # fold 内结果
        df_fold = fold_topk_to_df(topk_drug_id, topk_score, drug_id2bank, bank2name, exp_dis, explain_records_drug)
        fold_dfs.append(df_fold)

        # 写入 Excel
        df_fold.to_excel(
            writer,
            sheet_name=f"Fold_{cv+1}",
            index=False
        )

        # 保存用于跨 fold 平均
        for r, did, score in zip(
            range(1, len(topk_drug_id)+1),
            topk_drug_id,
            topk_score
        ):
            did = did.item()
            all_fold_results.append({
                "drug_id": did,
                "rank": r,
                "score": float(score)
            })

    ########################################
    # 4. rank-based ensemble
    ########################################

    df_all = pd.DataFrame(all_fold_results)

    summary = (
        df_all
        .groupby("drug_id")
        .agg(
            mean_rank=("rank", "mean"),
            mean_score=("score", "mean"),
            freq=("drug_id", "count")
        )
        .reset_index()
    )
    # 排序 & 取 TopK
    # summary = summary.sort_values("mean_rank").head(topk)
    summary = summary.sort_values(
        by=["freq", "mean_rank"],
        ascending=[False, True]
    )

    # 名称对齐
    summary["drugbank_id"] = summary["drug_id"].map(drug_id2bank)
    summary["drug_name"] = summary["drugbank_id"].map(bank2name)

    summary_df = summary[[
        "mean_rank",
        "drugbank_id",
        "drug_name",
        "mean_score",
        "freq"
    ]]

    # 写入第一个 sheet
    summary_df.to_excel(
        writer,
        sheet_name="Summary_MeanRank",
        index=False
    )

    top_summary_df, graph_summary_df, graph_detail_df = build_top_drug_graph_info(
        args=args,
        dataset=dataset,
        summary_df=summary_df,
        fold_dfs=fold_dfs,
        drug_id2bank=drug_id2bank,
        bank2name=bank2name,
        topn=args.graph_info_topn,
        fold=args.graph_info_fold,
    )
    top_summary_df.to_excel(
        writer,
        sheet_name=f"Top{args.graph_info_topn}_Fold{args.graph_info_fold}",
        index=False
    )
    graph_summary_df.to_excel(
        writer,
        sheet_name=f"GraphInfo_Summary_F{args.graph_info_fold}",
        index=False
    )
    graph_detail_df.to_excel(
        writer,
        sheet_name=f"GraphInfo_Detail_F{args.graph_info_fold}",
        index=False
    )
    disease_info_df, disease_graph_summary_df, disease_graph_detail_df = build_disease_graph_info(
        args=args,
        dataset=dataset,
        fold_dfs=fold_dfs,
        drug_id2bank=drug_id2bank,
        bank2name=bank2name,
        disease_id=args.graph_info_disease_id,
        fold=args.graph_info_fold,
    )
    disease_info_df.to_excel(
        writer,
        sheet_name=f"Disease_F{args.graph_info_fold}",
        index=False
    )
    disease_graph_summary_df.to_excel(
        writer,
        sheet_name=f"DiseaseGraph_Summary_F{args.graph_info_fold}",
        index=False
    )
    disease_graph_detail_df.to_excel(
        writer,
        sheet_name=f"DiseaseGraph_Detail_F{args.graph_info_fold}",
        index=False
    )
    ppi_status_df, ppi_overlap_neighbors_df, ppi_path_df, ppi_summary_df = build_single_drug_disease_ppi_info(
        args=args,
        dataset=dataset,
        top_summary_df=top_summary_df,
        drug_id2bank=drug_id2bank,
        bank2name=bank2name,
    )
    ppi_status_df.to_excel(
        writer,
        sheet_name=f"PPI_Status_F{args.graph_info_fold}",
        index=False
    )
    if not ppi_overlap_neighbors_df.empty:
        ppi_overlap_neighbors_df.to_excel(
            writer,
            sheet_name=f"PPI_Overlap3Hop_F{args.graph_info_fold}",
            index=False
        )
    if not ppi_path_df.empty:
        ppi_path_df.to_excel(
            writer,
            sheet_name=f"PPI_Path3Hop_F{args.graph_info_fold}",
            index=False
        )
    if not ppi_summary_df.empty:
        ppi_summary_df.to_excel(
            writer,
            sheet_name=f"PPI_ProteinSummary_F{args.graph_info_fold}",
            index=False
        )

    writer.close()

    print("======================================")
    print("10-fold AUC:", auc_list)
    print("10-fold AUPR:", aupr_list)
    print("Mean AUC:", np.mean(auc_list))
    print("Mean AUPR:", np.mean(aupr_list))
    print(f"Saved to {save_path}")
    print(f"Saved top-{args.graph_info_topn} graph info for Fold_{args.graph_info_fold}")
    print(f"Saved disease graph info for disease_id={args.graph_info_disease_id}, Fold_{args.graph_info_fold}")
    print(f"Saved single-drug PPI protein analysis for Fold_{args.graph_info_fold}")

    return auc_list, aupr_list, summary_df

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
    # parser.add_argument('--num_hidden', type=int, default=75)     # projection 的输入dim 和 输出 dim
    # parser.add_argument('--num_proj_hidden1', type=int, default=100)
    # parser.add_argument('--num_proj_hidden2', type=int, default=150)
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
    parser.add_argument('--graph_info_topn', type=int, default=10,
                        help='根据Summary_MeanRank导出前N个药物的图信息，默认10')
    parser.add_argument('--graph_info_fold', type=int, default=1,
                        help='导出哪个fold中的图信息，1-based，默认1')
    parser.add_argument('--graph_info_disease_id', type=int, default=2263,
                        help='导出指定疾病在几个图中的邻居信息，默认50')
    parser.add_argument('--graph_info_disease_name', type=str, default=None,
                        help='可选：按disease_name.csv中的疾病名指定疾病，例如MIMxxxxxx；设置后优先于graph_info_disease_id')
    parser.add_argument('--protein_path_top_id', type=int, default=1,
                        help='查看最终Summary_MeanRank排序TopK中的第几个药物，默认1；ID=1表示排序第一个药')
    # parser.add_argument('--alpha', default=0.05, type=float)
    # parser.add_argument('--reg', default=1e-3, type=float)

    args = parser.parse_args()
    args.graph_info_disease_id = resolve_graph_info_disease_id(
        args.data_name,
        args.graph_info_disease_id,
        args.graph_info_disease_name
    )
    print(args)
    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    aucs, auprs = [], []
    para_name = params_to_string(args)
    para_name = f"15-{para_name}"



    for times in range(0, 1):
        print("++++++++++++++++++times", str(times), "++++++++++++++++++++++")
        # args.save_dir = args.data_name + "_" + ''.join(str(times) + 'time')
        # args.save_dir = os.path.join("result", args.save_dir)
        #
        # if not os.path.isdir(args.save_dir):
        #     os.makedirs(args.save_dir)

        # file_path = args.save_dir + f"/{args.save_name}.xlsx"
        # if not os.path.exists(file_path):
        #     # 创建一个DataFrame
        #     columns = ["Parameter", "AUC", "AUPR", "AUC_list", "AUPR_list"]
        #     df = pd.DataFrame(columns=columns)
        #     df.to_excel(file_path, index=False)
        #     # 读取现有的Excel文件
        # df = pd.read_excel(file_path)

        args.model_path = os.path.join("weight", args.data_name + "_" + ''.join(str(times) + 'time'))
        if not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)

        dataset = DrugDataLoader(args, args.data_name, args.device,
                                 symm=args.gcn_agg_norm_symm,
                                 k=args.num_neighbor)

        print("Loading dataset finished ...\n")

        drug_id_map = pd.read_csv(f"./name_data/drug_data/{args.data_name}/drug_name.csv")

        drugbank_name_map = pd.read_csv(f"./name_data/drug_data/{args.data_name}/drugbank_drugs.csv")

        drugbank_name_map = drugbank_name_map.drop_duplicates(
            subset=["drugbank_id"],
            keep="first"
        )

        drug_id2bank = dict(zip(drug_id_map["drug_id"], drug_id_map["drug_name"]))
        bank2name = dict(zip(drugbank_name_map["drugbank_id"], drugbank_name_map["drug_name"]))
        OMIM_name = "FA"  # 智力发育障碍伴语言障碍和孤独症特征，常见对应FOXP1综合征等
        case_id = args.graph_info_disease_id
        case_list_id = [case_id]
        auc_list, aupr_list, summary_df = run_10fold_and_screen(
            args=args,
            para_name=para_name,
            dataset=dataset,
            drug_id2bank=drug_id2bank,
            bank2name=bank2name,
            topk=20,
            save_path=f"./case/{args.data_name}/{OMIM_name}_Tok7-graphinfo{str(args.protein_path_top_id)}-protein.xlsx",
            case_list_id = case_list_id
        )

        print(para_name)

