
import os
import json
import sys
import time
import re
# from random import choice
# import requests
from typing import List, Union, Dict
# from joblib import Parallel, delayed
import pickle

from pip._internal import network
from tqdm import  tqdm


def graph_profile(graph_info, score_type, entity_name, entity_type):
    if score_type == "dir_score":
        network_type = "药物-疾病治疗直接关联网络"
        relation_type = "已知或推断的治疗关联关系"
        weight_score = f"-在{entity_name} embedding 中的权重：{round(graph_info[score_type]['score_value'], 2) * 100}%"
    # elif score_type == "evi_score":
    #     network_type = "蛋白扰动网络"
    elif score_type == "pro_score":
        if entity_type == "Drug":
            network_type = "靶标蛋白-药物网络"
        else:
            network_type = "疾病-靶标蛋白网络"
        relation_type = "与靶标蛋白之间的关联关系"
        weight_score = (f"-在{entity_name} embedding 中的权重："
                        f"{round(graph_info['evi_score']['score_value'] * graph_info[score_type]['score_value'], 2) * 100}%")
    elif score_type == "sim_score":
        if entity_type == "Drug":
            network_type = "药物相似性网络"
            relation_type = "根据化学结构式、蛋白序列、蛋白通路、GO 生物过程、GO 分子功能、GO 细胞组分计算的相似性关联药物"
        else:
            network_type = "疾病相似性网络"
            relation_type = "根据KEGG表型注释、蛋白序列、蛋白通路、、GO 生物过程、GO 分子功能、GO 细胞组分计算的相似性关联疾病"
        weight_score = (f"-在{entity_name} embedding 中的权重："
                        f"{round(graph_info['evi_score']['score_value'] * graph_info[score_type]['score_value'],2) *100}%")
    else:
        print("Unknown score type!")
        sys.exit()
    target_node = entity_name
    if graph_info[score_type]['neighbor_count'] != 0:
        related_node = f"共{graph_info[score_type]['neighbor_count']}个相关节点：" + ", ".join(map(str,graph_info[score_type]["neighbors"]))
    else:
        related_node = "未找到相关邻居节点"

    profile = (f"-网络类型：{network_type}\n"
               f"-目标节点：{target_node}\n"
               f"-关联节点：{related_node}\n"
               f"-网络关系描述：{relation_type}\n"
               f"{weight_score}")
    return profile

def protein_profile(data):
    pro_profile = data["ppi_status"]["overlap_protein"]
    if "drug-protein和protein-disease图没有重合protein，但在PPI三阶内可达，已输出路径。" not in pro_profile:
        return pro_profile
    ppi_hop = data["ppi_hop1"]["protein_map"]
    if data["ppi_hop1"]["flag"] == "ppi_reachable":
        if len(ppi_hop["hop_1"]) > 0:
            pro_profile = pro_profile.replace("但在PPI三阶内可达，已输出路径。",
                                              f"但在PPI一阶可达，路径为：\n")
            protein_path = "".join(map(str, ppi_hop["hop_1"]))
            pro_profile = f"{pro_profile}{protein_path}"
            return pro_profile
        elif len(ppi_hop["hop_2"]) > 0:
            pro_profile = pro_profile.replace("但在PPI三阶内可达，已输出路径。",
                                              f"但在PPI二阶可达，路径为：\n")
            protein_path = "".join(map(str, ppi_hop["hop_2"]))
            pro_profile = f"{pro_profile}{protein_path}"
            return pro_profile
        else:
            pro_profile = pro_profile.replace("但在PPI三阶内可达，已输出路径。",
                                              f"但在PPI三阶可达，路径为：\n")
            protein_path = "".join(map(str, ppi_hop["hop_3"]))
            pro_profile = f"{pro_profile}{protein_path}"
            return pro_profile
    else:
        pro_profile = pro_profile.replace("但在PPI三阶内可达，已输出路径。", "在PPI也不可达。")
    return pro_profile


def check_explanation(model_output, network_profile, drug_name, disease_name, model_api):
    user_require_prompt = f'''
请判断下面这段药物重定位解释是否过关。

如果质量过关，只输出：
FLAG_PASS

如果质量不过关，请直接输出改进后的完整版本，不要输出原因、分析过程或其他多余内容。

【药物名称】
{drug_name}

【疾病名称】
{disease_name}

【原始网络证据】
    '''
    user_prompt = f"{user_require_prompt} \n {network_profile} \n 【待评价解释文本】\n {model_output}"
    model_output = model_api("", user_prompt, check_flag=True)
    return model_output



def choice_entity_pair(**kwargs):
    model_api = kwargs['model_api']
    start_num = kwargs['start_num']
    end_num = kwargs['end_num']
    data = kwargs['data']['example']
    keyword = kwargs['keyword']
    prompt = kwargs['prompt']
    save_directory = kwargs['save_directory']

    model_answer_dict = []
    for i in range(start_num, end_num):
        drug_name = data[i]["drug_name"]
        disease_name = data[i]["disease_name"]
        score_list = ["dir_score", "pro_score", "sim_score"]

        drug_graph_info = data[i]["graph_info_summary"]  # dict
        drug_profile = (f"{drug_name}的表示来源自两个网络视角：\n"
                   f"其中，{round(drug_graph_info['dir_score']['score_value'],2) * 100}%来自于视角1，即药物-疾病治疗直接关联网络；"
                        f"有{round(drug_graph_info['evi_score']['score_value'],2) * 100}%来自于视角2，即蛋白扰动间接关联网络，由靶标蛋白-药物网络和药物相似性网络组成。\n"
                        f"具体网络信息如下：")

        dis_graph_info = data[i]["disease_graph_summary"]["scores"]  # dict
        dis_profile = (f"{disease_name}的表示来源自两个网络视角：\n"
                        f"其中，{round(dis_graph_info['dir_score']['score_value'], 2) * 100}%来自于视角1，即药物-疾病治疗直接关联网络；"
                        f"有{round(dis_graph_info['evi_score']['score_value'], 2) * 100}%来自于视角2，即蛋白扰动间接关联网络，由靶标蛋白-药物网络和药物相似性网络组成。\n"
                       f"具体网络信息如下：")
        for score_type in score_list:
            drug_network_profile = graph_profile(graph_info=drug_graph_info,
                                                 score_type=score_type,
                                                 entity_name=drug_name,
                                                 entity_type="Drug")
            drug_profile = f"{drug_profile} \n {drug_network_profile} \n"
            dis_network_profile = graph_profile(graph_info=dis_graph_info,
                                                score_type=score_type,
                                                entity_name=disease_name,
                                                entity_type="Disease")
            dis_profile = f"{dis_profile} \n {dis_network_profile} \n"
        pro_profile = protein_profile(data[i])
        # sys_prompt = prompt.replace("<herb_name>", drug_name).replace("<symptom_name>", disease_name)
        sys_prompt = prompt
        user_require_prompt = '''
请输出一段不超过800字的中文学术性解释，包括：

1. 总体判断：说明该预测主要由哪些网络视角支持；
2. 主要证据来源：分别概括药物侧、疾病侧和桥接证据；
3. 最佳作用路径排序：按网络权重和连接强度排序，优先解释 protein-PPI 路径；
4. 机制假说：说明药物 <drug_name> 可能通过其相关蛋白扰动，与疾病 <disease_name> 相关蛋白模块产生间接影响；

注意：
- 不得引入未给出的蛋白、通路或外部机制；
- 不得声称该药已经被证实可治疗该疾病；
- 不要简单罗列所有节点，应围绕权重最高、连接最清晰的证据展开。

以下是药物 <drug_name> 和 疾病 <disease_name> 的网络证据：
        '''
        network_profile = f"药物{drug_profile} \n 疾病{dis_profile} 上述疾病和药物相关的蛋白互作子网络信息如下：\n {pro_profile}"
        user_prompt = f"{user_require_prompt} \n {network_profile}"
        model_output = model_api(sys_prompt, user_prompt, check_flag=False)
        max_check_rounds = 5

        for index in range(max_check_rounds):
            check_flag = check_explanation(
                model_output,
                network_profile,
                drug_name,
                disease_name,
                model_api
            )
            if check_flag == "FLAG_PASS":
                print(index)
                print(check_flag)
                break
            # check_flag 是改进后的文本
            model_output = check_flag
        else:
            print("Warning: explanation did not pass check after max rounds.")


        dict = {
            'drug_name': drug_name,
            'disease_name': disease_name,
            'explain_graph': model_output
        }
        # print("*" * 100, "index-", dict["index"], "*" * 100)
        for key, value in dict.items():
            print(key, ":", value)
        # print(dict)
        model_answer_dict.append(dict)

        file_name = f"{disease_name}-{drug_name}_exp.json"
        file_path = os.path.join(save_directory, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            output = {
                'keyword': keyword,
                'example': model_answer_dict
            }
            json.dump(output, f, ensure_ascii=False, indent=4)
            f.close()


def export_union_json(directory: str, model_name: str, keyword: str, zero_shot_prompt_text: str) -> None:
    """
    Merges JSON files containing processed examples in a directory into a single JSON file.

    :param directory: Directory containing the JSON files
    :param model_name: Name of the model used to process the examples
    :param keyword: Keyword used to identify the JSON files
    :param zero_shot_prompt_text: Prompt text for zero-shot learning
    :param question_type: Type of questions in the JSON files (e.g. single_choice, five_out_of_seven, etc.)
    """

    save_directory = os.path.join(directory, f'{model_name}_{keyword}_summary')
    # save_directory = os.path.join(directory, f'{model_name}_{keyword}')  # herb_pair
    if os.path.exists(save_directory):
        output = {
                        'keyword': keyword,
                        'model_name': model_name,
                        'prompt': zero_shot_prompt_text,
                        'example': []
                    }

        # Iterate through the JSON files with the specified keyword in the directory

        print("Start to merge json files")
        files = [file for file in os.listdir(save_directory) if file.endswith('.json')]
        # print("Start to merge json files")
        example_num = len(files)
        parallel_num = example_num
        batch_size = example_num / parallel_num
        for idx in range(0, parallel_num):
            # print("*" * 100, idx)
            start_num = idx * batch_size
            end_num = min(start_num + batch_size, example_num)
            if start_num >= example_num:
                break
            file_name = f"seperate_{int(start_num)}-{int(end_num-1)}.json"
            file_path = os.path.join(save_directory, file_name)
            # Load and merge the data from the JSON files
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                output['example'] += (data['example'])

        # Save the merged data into a single JSON file
        # merge_file = os.path.join(directory, f'{model_name}_{keyword}.json') # herb_pair
        merge_file = os.path.join(directory, f'{model_name}_{keyword}_summary.json')
        # output['example'] = sorted(output['example'], key=lambda x: x['index'])
        with open(merge_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            print(merge_file)
        print(merge_file)

def export_distribute_json(
        model_api,
        model_name: str, 
        directory: str, 
        keyword: str, 
        zero_shot_prompt_text: str or List[str],
        file_name: str
    ) -> None:
    """
    Distributes the task of processing examples in a JSON file across multiple processes.

    :param model_name: Name of the model to use
    :param directory: Directory containing the JSON file
    :param keyword: Keyword used to identify the JSON file
    :param zero_shot_prompt_text: Prompt text for zero-shot learning
    """
    # Find the JSON file with the specified keyword
    for root, _, files in os.walk(directory):
        for file in files:
            if file == file_name:
            # if file == f'{keyword}.json':   # herb_pair
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
    
    example_num = len(data['example'])
    parallel_num = example_num
        
    # Prepare the list of keyword arguments for parallel processing
    kwargs_list = []
    batch_size = example_num // parallel_num
    save_directory = os.path.join(directory, f'{model_name}_{keyword}')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for idx in range(0, parallel_num):
        start_num = idx * batch_size
        end_num = min(start_num + batch_size, example_num)
        if start_num >= example_num:
            break

        kwargs = {
            'model_api': model_api,
            'start_num': start_num,
            'end_num': end_num,
            'model_name': model_name, 
            'data': data, 
            'keyword': keyword, 
            'prompt': zero_shot_prompt_text,
            'save_directory': save_directory,
        }
        kwargs_list.append(kwargs)
    
    # Run parallel processing based on the question type
    for kwargs in kwargs_list:
        choice_entity_pair(**kwargs)
