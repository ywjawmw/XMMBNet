import random
import sys

import requests
import time
import openai
import os
from random import choice


class OpenaiAPI:
    def __init__(self, api_key_list: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0, max_tokens: int = 1024):
        self.api_key_list = api_key_list
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def send_request_entity_pair(self, sys_prompt, user_message):
        """
        """
        messages = []
        # sys_prompt = sensitive(sys_prompt)
        zero_shot_prompt_message = {"role": "system", "content": sys_prompt}
        messages.append(zero_shot_prompt_message)
        # user_message = f"{sys_prompt}\n{user_message}"
        # user_message = sensitive(user_message)
        print("sys_prompt:", sys_prompt)
        message = {"role": "user", "content": user_message}
        print(f"LLM的Prompt是{'*' * 100}\n{message['content']}")
        messages.append(message)
        while True:
            try:
                os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
                openai.api_key = self.api_key_list
                openai.api_base = "https://xiaoai.plus/v1"
                output = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature
                )
                # output.choices[0].message.content = revers_sensitive(output.choices[0].message.content)
                answer = output.choices[0].message.content
                # answer = revers_sensitive(answer)
                # print(answer)
                return answer
            except Exception as e:
                print("Exception:", e)
                print("原始Prompt：")
                sys.exit()

    def check_quality(self, user_message):
        messages = []
        sys_prompt = '''
你是一名药物重定位、生物网络分析和科学文本审校专家。你的任务是判断一段药物-疾病重定位机制解释是否充分、准确、严谨，并在必要时进行改写。

请严格遵守以下规则：

1. 只能基于用户提供的原始网络证据和待评价文本进行判断或改写，不得引入未出现的蛋白、药物、疾病、通路、机制或外部知识。
2. 重点检查以下质量标准：
   - 是否正确体现药物侧和疾病侧各网络视角的贡献权重；
   - 是否同时分析药物侧证据、疾病侧证据和药物-疾病桥接证据；
   - 是否正确说明 drug-protein 与 disease-protein 没有共同蛋白；
   - 是否避免将 PPI 间接连接写成直接靶点重合；
   - 是否避免夸大为已证实疗效；
   - 是否存在权重排序错误、节点遗漏导致的关键证据缺失、外部机制引入或因果表述过强；
   - 是否语言简洁、客观、学术化。
3. 如果待评价文本满足质量标准，只输出：
FLAG_PASS
4. 如果待评价文本不满足质量标准，不要输出失败原因，不要输出分析过程，直接输出改进后的完整解释文本。
5. 改进后的文本应不超过800字，和原始解释结构完全一致，结构包括：
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

6. 改写时不得机械罗列所有节点，应围绕权重最高、连接最清晰的证据展开。
7. 最终输出只能有两种形式：
形式A：
FLAG_PASS

形式B：
【总体判断】
【主要证据来源】
【最佳作用路径排序】
【机制性解释】
禁止输出“质量不过关”，“修改如下，“原因是”等任何过渡语。
        '''
        zero_shot_prompt_message = {"role": "system", "content": sys_prompt}
        messages.append(zero_shot_prompt_message)
        # user_message = f"{sys_prompt}\n{user_message}"
        # user_message = sensitive(user_message)
        print("sys_prompt:", sys_prompt)
        message = {"role": "user", "content": user_message}
        print(f"LLM的Prompt是{'*' * 100}\n{message['content']}")
        messages.append(message)
        while True:
            try:
                os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
                openai.api_key = self.api_key_list
                openai.api_base = "https://xiaoai.plus/v1"
                output = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature
                )
                # output.choices[0].message.content = revers_sensitive(output.choices[0].message.content)
                answer = output.choices[0].message.content
                # answer = revers_sensitive(answer)
                # print(answer)
                return answer
            except Exception as e:
                print("Exception:", e)
                print("原始Prompt：")
                sys.exit()

    def forward(self, sys_prompt, user_message, check_flag=False) -> list:
        """
        """
        if check_flag:
            output = self.check_quality(user_message)
        else:
            output = self.send_request_entity_pair(sys_prompt, user_message)
        return output

    def postprocess(self, output):
        """
        """
        model_output = None
        try:
            if "gpt" in self.model_name:
                model_output = output["choices"][0]["message"]["content"]

            elif self.model_name == "text-davinci-003":
                model_output = output["choices"][0]["text"]

            if not model_output:
                print("Warning: Empty Output ")
        except Exception as e:
            print("Exception:", e)
            model_output = "【解析】\n"
            print("Warning error: Empty Output ")
        return model_output

    def __call__(self, sys_prompt: str, user_message: str, check_flag: bool):
        return self.forward(sys_prompt, user_message, check_flag)


# def sensitive(sentence):
#     sentence = sentence.replace("阴道", "term-YD")
#     sentence = sentence.replace("射精", "term-SJ")
#     return sentence
#
# def revers_sensitive(sentence):
#     sentence = sentence.replace("term-YD", "阴道")
#     sentence = sentence.replace("term-SJ", "射精")
#     return sentence
