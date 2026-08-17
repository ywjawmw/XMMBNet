# --*-- conding:utf-8 --*--
# @Time : 2026/5/26 15:46
# @Author : YWJ
# @Email :
# @File : get_LLM_response.py
# @Software : PyCharm
# @Description : 根据graph_info_ppi_path.json的内容获得LLM的回复
import json
from Openai import OpenaiAPI
# from Qwen import QwenAPI
from bench_function import export_distribute_json, export_union_json
import os



# explain_prompt = '''
# 你是一名药物重定位的研究专家，协助我来从作用机制的角度，解析药物为什么会对疾病起到治疗作用。\n
# 以下信息来自一个多视角生物网络融合模型的推理结果，每一条表示药物 "<herb_name>" 和 疾病 "<symptom_name>" 在进行关系预测时，对应网络中的证据支持强度，要充分理解并分别分析药物和疾病两类实体的相关网络信息，然后按照要求进行总结和分析。请输出一段简洁、通顺、学术性的文字，来解释药物 "<herb_name>" 被重定位为治疗 疾病 "<symptom_name>" 的证据。 \n
# 输出的请严格遵循以下规则： \n
# 1.仅基于下述给出的网络信息进行解释，不得引入未出现的蛋白、通路或机制； \n
# 2.输出的内容应该容易理解、简洁简短、客观、内容正确的生物作用机制解析，不要超过1000个单词。 \n
# 3.若某类网络中明确说明“未找到相关邻居节点”，请如实反映该不确定性，不得补全； \n
# 4. 在解释中体现各类网络对预测的相对贡献权重； \n
# 5. 输出应包括： \n
# （1）主要证据来源，提供的所有蛋白、疾病和药物都不要省略。 \n
# （2）最佳作用路径，并根据网络权重进行排序，并给出排序理论。 具体信息如下：
# '''

explain_prompt = '''
你是一名药物重定位与生物网络分析专家，任务是基于给定的多视角生物网络证据，解释某个药物为什么可能被重定位用于某种疾病。

请严格遵守以下原则：
1. 只能使用用户提供的网络信息进行解释，不得引入未给出的蛋白、通路、疾病、药物、临床结论或外部知识。
2. 输出应被表述为“网络模型支持的机制假说”，不得表述为已经证实的临床疗效。
3. 需要同时分析药物侧证据、疾病侧证据，以及二者之间的桥接证据；不能只根据药物侧邻居得出重定位解释。
4. 若 drug-protein 网络与 disease-protein 网络没有共同蛋白，但存在PPI可达路径，应明确说明这是“间接蛋白互作连接”，不得写成直接靶点重合。
5. 解释各网络视角的相对贡献权重，并根据权重判断主要证据来源。
6. 对“直接治疗关联网络”、“相似性网络”和“蛋白关联网络”要区分解释：
   - 直接治疗关联网络用于说明已知或推断的疾病/药物邻域背景；
   - 相似性网络用于说明相似药物或相似疾病带来的辅助证据；
   - 蛋白关联网络和 PPI 路径用于说明潜在分子作用机制。
7. 如果某类网络说明“未找到相关邻居节点”，必须如实报告不确定性，不得补全。
8. 不要机械堆砌节点名称。正文中优先解释关键证据和关键路径；完整节点可在“证据节点列表”中简要列出。
9. 输出应客观、简洁、学术化，避免夸大因果关系。

输出格式如下：

【总体判断】
用2-4句话概括该药物被预测为治疗该疾病的主要网络依据。

【主要证据来源】
分别说明：
1. 药物侧证据；
2. 疾病侧证据；
3. 药物-疾病桥接证据；
并标明各视角权重。

【最佳作用路径排序】
按证据强度排序列出若干条路径。每条路径需包括：
- 头实体要为疾病，尾实体要为药物
- 路径类型；
- 具体节点；
- 涉及的网络视角和权重；
- 为什么该路径排在该位置。

【机制性解释】
整合上述证据，形成一段连贯的机制假说。
'''


# 读写json文件
def read_json(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    return data

def write_json(file_path, data, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# 下面是头痛与其他症状之间的信息：

if __name__ == "__main__":
    # Load the FBQ_prompt.json file
    directory = "./graph_information"
    # get the model_name and instantiate model_api
    model_type = "OpenAI"
    if model_type == "OpenAI":
        os.environ['HTTPS_PROXY'] = "http://127.0.0.1:XXX"
        openai_api_key = "XXX"  # your key gpt-5
        os.environ['OPENAI_API_KEY'] = openai_api_key
        os.environ["OPENAI_BASE_URL"] = "XXX"
        # model_name = 'gpt-4-0613'
        # model_name = 'gpt-4o-2024-05-13'
        model_name = 'gpt-5.5'
        model_api = OpenaiAPI(openai_api_key, model_name=model_name)
    elif model_type == "Qwen":
        os.environ['HTTPS_PROXY'] = "http://127.0.0.1:XXX"
        # model_name = "qwen-max"
        # model_name = "qwen2.5-7b-instruct-1m"
        model_name = "deepseek-r1"
        model_api = QwenAPI(model_name=model_name)

    # question_type = "entity_pair" # "entity_pair"
    # print(model_name)
    # print(question_type)

    file_name = "FA_graph_info_ppi_path_top9_fold2.json"

    keyword = "explain_DDA"
    zero_shot_prompt_text = explain_prompt
    print(keyword)

    export_distribute_json(
        model_api,
        model_name,
        directory,
        keyword,
        zero_shot_prompt_text,
        file_name
        # question_type,  # "entity_pair"、encoder
        # entity_type="herb",
    )

    export_union_json(
        directory,
        model_name,
        keyword,
        zero_shot_prompt_text
    )

'''
--disease_name MIM145500 --setting xr-view1 --para_prefix 15-xr --train_lr 1e-3 --times 0
--disease_name MIM145500 --setting best --para_prefix 15 --train_lr 3e-2 --times 1 
'''