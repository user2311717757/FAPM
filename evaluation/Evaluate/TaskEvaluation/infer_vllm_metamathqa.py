import os
import sys
import json
import pdb
import random
import torch
import re
from fraction import Fraction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rouge import Rouge

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

model_path = "/mnt/data2/nianke_catastrophic_forgetting/LLaMA-Factory-main/saves/Llama3-lora/merge/Llama3-metamathqa-5e-6"
#/mnt/data1/model/meta-llama/Meta-Llama-3-8B-Instruct
# print(data_type, model_path)

from swift.llm import (
    ModelType,
    get_vllm_engine,
    get_default_template_type,
    get_template,
    inference_vllm,
)

model_type = ModelType.llama3_8b_instruct

llm_engine = get_vllm_engine(
    model_type,
    torch.bfloat16,
    model_id_or_path=model_path,
    gpu_memory_utilization=0.95,
    tensor_parallel_size=1,
)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_tokens = 512
# llm_engine.generation_config.repetition_penalty = 1.1
llm_engine.generation_config.temperature = 0.0
llm_engine.generation_config.top_k = 20
# llm_engine.generation_config.top_p = 1.0
# llm_engine.generation_config.seed = 4
print(llm_engine.generation_config)

data = []
with open("/mnt/data1/nianke_catastrophic_forgetting/LLaMA-Factory-main/data_metamathqa/GSM8K_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))


data_infer = []
label_list = []

inst = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"

for idx,element in enumerate(data):
    dict1 = {}
    dict1['query'] = inst + element['query'] + "\n\n### Response:"
    data_infer.append(dict1)

    temp_ans = element['response'].split('#### ')[1]
    temp_ans = int(temp_ans.replace(',', ''))
    label_list.append(temp_ans)


resp_list = inference_vllm(llm_engine, template, data_infer, use_tqdm=True)

result = []
invalid = 0
for index, response in enumerate(resp_list):
    # res = response["response"].strip().lower()
    res = response["response"].strip()
    y_pred = extract_answer_number(res)
    result.append(y_pred)

if len(result) != len(label_list):
        print("error")

i = 0
for ans in range(len(result)):
    if result[ans] == label_list[ans]:
        i = i + 1
print(i/len(result))