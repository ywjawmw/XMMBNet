import torch as th
from sklearn import metrics
import dgl
import pandas as pd
import numpy as np

def build_infer_graph_for_one_disease(
    disease_id: int,
    num_drug: int,
    num_disease: int,
    device
):
    # 所有 drug
    drug_ids = th.arange(num_drug, device=device)

    # disease 重复 num_drug 次
    disease_ids = th.full(
        (num_drug,),
        disease_id,
        dtype=th.long,
        device=device
    )

    graph = dgl.heterograph(
        {
            ('drug', 'rate', 'disease'): (drug_ids, disease_ids)
        },
        num_nodes_dict={
            'drug': num_drug,
            'disease': num_disease
        }
    ).to(device)

    return graph

def rank_drugs_for_disease(
    disease_id,
    drug_feat,
    dis_feat,
    decoder,
    num_drug,
    num_disease,
    topk=20,
    device='cuda'
):
    decoder.eval()

    graph = build_infer_graph_for_one_disease(
        disease_id, num_drug, num_disease, device
    )

    scores = decoder(
        graph,
        drug_feat.to(device),
        dis_feat.to(device)
    ).squeeze()   # [num_drug]

    topk_scores, topk_drug_ids = th.topk(scores, topk)

    return topk_drug_ids, topk_scores


def screen_drugs_on_negative_only(
    drug_dis_csv,
    dis_id,
    drug_feat,
    dis_feat,
    decoder,
    num_drug,
    num_disease,
    topk=20,
    device='cuda'
):
    """
    仅在负样本(值=0)药物中进行筛选
    """

    # 1. 读取 drug-disease 矩阵
    mat = pd.read_csv(drug_dis_csv, index_col=0)  # [num_drug, num_disease]

    # 2. 找到该疾病下的负样本 drug
    neg_drug_ids = np.where(mat.iloc[:, dis_id].values == 0)[0]
    neg_drug_ids = th.tensor(neg_drug_ids, dtype=th.long, device=device)

    # 3. 构造推理用 bipartite graph（不需要标签）
    dis_ids = th.full_like(neg_drug_ids, dis_id)

    infer_graph = dgl.heterograph(
        {('drug', 'rate', 'disease'): (neg_drug_ids, dis_ids)},
        num_nodes_dict={
            'drug': num_drug,
            'disease': num_disease
        }
    ).to(device)

    # 4. 用训练好的 MLPDecoder 打分
    with th.no_grad():
        scores = decoder(
            infer_graph,
            drug_feat.to(device),
            dis_feat.to(device)
        ).squeeze()   # [num_neg_drug]
        scores = minmax_norm(scores)

    # 5. 取 Top-K
    topk_idx = th.topk(scores, k=min(topk, scores.shape[0])).indices
    topk_drug_ids = neg_drug_ids[topk_idx]
    topk_scores = scores[topk_idx]

    return topk_drug_ids, topk_scores

def minmax_norm(x, eps=1e-8):
    return (x - x.min()) / (x.max() - x.min() + eps)

####可解释性权重，只针对视角内，因为有三个约束
def check_isolated_node(graph, node_type, device):
    """
    判断某一类节点在图中是否为孤立节点
    返回: BoolTensor [N]
    """
    if node_type == "drug":
        in_deg = graph['rev-1'].in_degrees().to(device)
        out_deg = graph['1'].out_degrees().to(device)
    elif node_type == "disease":
        in_deg = graph['1'].in_degrees().to(device)
        out_deg = graph['rev-1'].out_degrees().to(device)
    else:
        raise ValueError("Unsupported node type")

    return (in_deg + out_deg) == 0

def extract_gate_weight(alpha_tensor, idx, iso=False):
    """
    alpha_tensor: [N, 2, 1]
    idx: node id
    return: dict
    """
    if iso:
        return {
            "view_0": (1.0+0.5)/2,
            "view_1": 0.5/2
        }
    else:
        return {
            "view_0": (float(alpha_tensor[idx, 0, 0].item()) + 0.5)/2,
            "view_1": (float(alpha_tensor[idx, 1, 0].item()) + 0.5)/2
        }

def build_drug_explanation(
    drug_ids,
    drug_scores,
    drug_beta,          # inner gate: [N,2,1]
    drug_gama,          # intra gate: [N,2,1]
    drug_pro_graph,
    node_type,
    device
):
    """
    返回 List[Dict]，每个 dict 对应一个 drug
    """
    isolated_mask = check_isolated_node(
        drug_pro_graph,
        node_type=node_type,
        device=device
    )

    results = []

    for rank, (drug_id, score) in enumerate(zip(drug_ids, drug_scores), start=1):
        is_isolated = bool(isolated_mask[drug_id].item())

        record = {
            "rank": rank,
            f"{node_type}_id": int(drug_id),
            "score": float(score),
            "is_isolated_in_DP_PS": is_isolated,
        }

        if not is_isolated:
            record["inner_view_gate"] = extract_gate_weight(drug_beta, drug_id, is_isolated)
            record["intra_view_gate"] = extract_gate_weight(drug_gama, drug_id)
            record["gate_valid"] = True
        else:
            record["inner_view_gate"] = extract_gate_weight(drug_beta, drug_id, is_isolated)
            record["intra_view_gate"] = extract_gate_weight(drug_gama, drug_id)
            record["gate_valid"] = False

        results.append(record)

    return results

def evaluate_GCL(args, model, graph_data,
                 drug_graph, drug_feat, drug_sim_feat,
                 dis_graph, dis_feat, dis_sim_feat,
                 pro_graph, pro_sim_feat,
                 drug_evi_feat, dis_evi_feat, pro_feat,
                 drug_pro_graph, pro_dis_graph, case_list_id=[439]):
    # if case_list_id is None:
    #     case_list_id = [439]
    rating_values = graph_data['test'][2]
    enc_graph = graph_data['test'][0].int().to(args.device)
    dec_graph = graph_data['test'][1].int().to(args.device)

    model.eval()
    # case_list_id = [1369, 50, 765, 301]
    # case_list_id = [50]   # RXA
    # case_list_id = [1369]  # YXA
    # case_list_id = [439]  # FA
    # case_list_id = [1514]  #  备选 ID=5 阿尔兹海默症 1514
    with th.no_grad():
        pred_ratings, drug_emb, dis_emb, weight_list = model(enc_graph, dec_graph,
                                                  drug_graph, drug_sim_feat, drug_feat,
                                                  dis_graph, dis_sim_feat, dis_feat,
                                                  pro_graph, pro_sim_feat,
                                                  drug_evi_feat, dis_evi_feat, pro_feat,
                                                  drug_pro_graph, pro_dis_graph, Two_Stage=False, train=False)
        drug_beta, dis_beta, drug_gama, dis_gama = weight_list[0], weight_list[1], weight_list[2], weight_list[3]
    y_score = pred_ratings.view(-1).cpu().tolist()
    y_true = rating_values.cpu().tolist()
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)

    topk_drug_id, topk_score = [], []

    for dis_case_id in case_list_id:
        # # 富集分析
        # topk_drug_id, topk_score = rank_drugs_for_disease(dis_case_id, drug_emb, dis_emb, model.decoder, num_drug=1220, num_disease=2480, topk=20)
        # 未知药筛选
        drug_dis_csv = "./name_data/drug_data/Adataset/drug_dis.csv"
        topk_drug_id, topk_score = screen_drugs_on_negative_only(
            drug_dis_csv, dis_case_id,
            drug_emb, dis_emb, model.decoder, num_drug=1220, num_disease=2480, topk=20
        )
        print(f"\nDisease ID: {dis_case_id}")

        explain_records_dis = build_drug_explanation(
            drug_ids=case_list_id,
            drug_scores=[0],
            drug_beta=dis_beta,  # fuse_drug_inner 输出
            drug_gama=dis_gama,  # fuse_drug_intra 输出
            drug_pro_graph=pro_dis_graph,
            node_type= "disease",
            device=args.device
        )
        exp_dis = {
            "sim_score": explain_records_dis[0]['inner_view_gate']['view_0'],
            "pro_score": explain_records_dis[0]['inner_view_gate']['view_1'],
            "dir_score": explain_records_dis[0]['intra_view_gate']['view_0'],
            "evi_score": explain_records_dis[0]['intra_view_gate']['view_1'],
        }
        print(exp_dis)
        # for rank, (drug_id, s) in enumerate(zip(topk_drug_id.tolist(), topk_score.tolist()), start=1):
        #     print(f"Rank {rank:02d}: Drug ID {drug_id}, score={s:.6f}")
        explain_records = build_drug_explanation(
            drug_ids=topk_drug_id,
            drug_scores=topk_score,
            drug_beta=drug_beta,  # fuse_drug_inner 输出
            drug_gama=drug_gama,  # fuse_drug_intra 输出
            drug_pro_graph=drug_pro_graph,
            node_type="drug",
            device=args.device
        )
        explain_records_drug = {}
        for rec in explain_records:
            print(rec)
            explain_records_drug[rec["drug_id"]]={
                    "sim_score": rec['inner_view_gate']['view_0'],
                    "pro_score": rec['inner_view_gate']['view_1'],
                    "dir_score": rec['intra_view_gate']['view_0'],
                    "evi_score": rec['intra_view_gate']['view_1'],
                }
    return auc, aupr, y_true, y_score, topk_drug_id, topk_score, exp_dis, explain_records_drug



def evaluate_GCL_fuji(args, model, graph_data,
                 drug_graph, drug_feat, drug_sim_feat,
                 dis_graph, dis_feat, dis_sim_feat,
                 pro_graph, pro_sim_feat,
                 drug_evi_feat, dis_evi_feat, pro_feat,
                 drug_pro_graph, pro_dis_graph, case_list_id=[439]):
    # if case_list_id is None:
    #     case_list_id = [439]
    rating_values = graph_data['test'][2]
    enc_graph = graph_data['test'][0].int().to(args.device)
    dec_graph = graph_data['test'][1].int().to(args.device)

    model.eval()
    # case_list_id = [1369, 50, 765, 301]
    # case_list_id = [50]   # RXA
    # case_list_id = [1369]  # YXA
    # case_list_id = [439]  # FA
    # case_list_id = [1514]  #  备选 ID=5 阿尔兹海默症 1514
    with th.no_grad():
        pred_ratings, drug_emb, dis_emb, weight_list = model(enc_graph, dec_graph,
                                                  drug_graph, drug_sim_feat, drug_feat,
                                                  dis_graph, dis_sim_feat, dis_feat,
                                                  pro_graph, pro_sim_feat,
                                                  drug_evi_feat, dis_evi_feat, pro_feat,
                                                  drug_pro_graph, pro_dis_graph, Two_Stage=False, train=False)
        drug_beta, dis_beta, drug_gama, dis_gama = weight_list[0], weight_list[1], weight_list[2], weight_list[3]
    y_score = pred_ratings.view(-1).cpu().tolist()
    y_true = rating_values.cpu().tolist()
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)

    topk_drug_id, topk_score = [], []

    for dis_case_id in case_list_id:
        # 富集分析
        topk_drug_id, topk_score = rank_drugs_for_disease(dis_case_id, drug_emb, dis_emb, model.decoder, num_drug=1220, num_disease=2480, topk=1220)
        # # 未知药筛选
        # drug_dis_csv = "./name_data/drug_data/Adataset/drug_dis.csv"
        # topk_drug_id, topk_score = screen_drugs_on_negative_only(
        #     drug_dis_csv, dis_case_id,
        #     drug_emb, dis_emb, model.decoder, num_drug=1220, num_disease=2480, topk=20
        # )
        print(f"\nDisease ID: {dis_case_id}")
        exp_dis = {}
        explain_records_drug = {}
        continue

        explain_records_dis = build_drug_explanation(
            drug_ids=case_list_id,
            drug_scores=[0],
            drug_beta=dis_beta,  # fuse_drug_inner 输出
            drug_gama=dis_gama,  # fuse_drug_intra 输出
            drug_pro_graph=pro_dis_graph,
            node_type= "disease",
            device=args.device
        )
        exp_dis = {
            "sim_score": explain_records_dis[0]['inner_view_gate']['view_0'],
            "pro_score": explain_records_dis[0]['inner_view_gate']['view_1'],
            "dir_score": explain_records_dis[0]['intra_view_gate']['view_0'],
            "evi_score": explain_records_dis[0]['intra_view_gate']['view_1'],
        }
        print(exp_dis)
        # for rank, (drug_id, s) in enumerate(zip(topk_drug_id.tolist(), topk_score.tolist()), start=1):
        #     print(f"Rank {rank:02d}: Drug ID {drug_id}, score={s:.6f}")
        explain_records = build_drug_explanation(
            drug_ids=topk_drug_id,
            drug_scores=topk_score,
            drug_beta=drug_beta,  # fuse_drug_inner 输出
            drug_gama=drug_gama,  # fuse_drug_intra 输出
            drug_pro_graph=drug_pro_graph,
            node_type="drug",
            device=args.device
        )
        explain_records_drug = {}
        for rec in explain_records:
            print(rec)
            explain_records_drug[rec["drug_id"]]={
                    "sim_score": rec['inner_view_gate']['view_0'],
                    "pro_score": rec['inner_view_gate']['view_1'],
                    "dir_score": rec['intra_view_gate']['view_0'],
                    "evi_score": rec['intra_view_gate']['view_1'],
                }
    return auc, aupr, y_true, y_score, topk_drug_id, topk_score, exp_dis, explain_records_drug
