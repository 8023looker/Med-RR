" Qwen-72B Rewritten Query "
import os
import ujson

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ujson
from tqdm import tqdm
import subprocess

import prompts

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


class QwenInfer:
    def __init__(self, model_path):
        self.llm = LLM(model=model_path, tensor_parallel_size=4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
        
    def prompted_infer_category(self, jsonl_line_content, query_type, language="en"):
        raw_text = jsonl_line_content["question"]
        
        prompt = prompts.classification_dict[query_type] # still in dict format
        if language == "zh": # Chinese
            prompt = prompt["zh"] + "\n\n" + raw_text
        else: # English
            prompt = prompt["en"] + "\n\n" + raw_text
        
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # generate outputs
        outputs = self.llm.generate([text], self.sampling_params)
        response = outputs[0].outputs[0].text
        # print(f"Prompt: {prompt!r}, Generated text: {response!r}")
        print(f"Generated text: {response!r}")

        # content[query_type] = self.translate_category(response, query_type, language) # add category key
        category = self.translate_category(response, query_type, language) # add category key
        return category

             
    def translate_category(self, topic, query_type, language):
        topic = topic.lower() # lowercase
        
        category_trans_dict = {
            "EBM_classification": {
                "zh": {
                    "预后": "prognosis",
                    "治疗": "therapy",
                    "病因": "etiology",
                    "诊断": "diagnosis",
                    "预防": "prevention",
                    "成本": "cost",
                    "其他": "others",
                    "其它": "others"
                },
                "en": {
                    "prognosis": "prognosis",
                    "therapy": "therapy",
                    "etiology": "etiology",
                    "diagnosis": "diagnosis",
                    "prevention": "prevention",
                    "cost": "cost",
                    "others": "others",
                    "other": "others"
                }
            },
            "question_classification": {
                "zh": {
                    "事实型": "factual",
                    "定义型": "definitional",
                    "解释型": "explanatory",
                    "描述型": "descriptive",
                    "指示型": "directive",
                    "意见型": "opinion",
                    "比较型": "comparative",
                    "评价型": "evaluative",
                    "假设型": "hypothetical",
                    "程序型": "procedural",
                    "参考型": "referential",
                    "验证型": "verification",
                    "其他类型": "other",
                    "其他": "others",
                    "其它": "others"
                    
                },
                "en": {
                    "factual": "factual",
                    "definitional": "definitional",
                    "explanatory": "explanatory",
                    "descriptive": "descriptive",
                    "directive": "directive",
                    "opinion": "opinion",
                    "comparative": "comparative",
                    "evaluative": "evaluative",
                    "hypothetical": "hypothetical",
                    "procedural": "procedural",
                    "referential": "referential",
                    "verification": "verification",
                    "other": "others",
                    "others": "others"
                }
            }
        }
        
        if topic not in category_trans_dict[query_type][language]:
            return f"error-{topic}"
        else:
            return category_trans_dict[query_type][language][topic]
    
     
    def query_rewriting(self, jsonl_line_content, EBM_type, language="en"): # 传入的是已经解析好的 jsonl_line_content
        raw_text = jsonl_line_content["question"]
        
        prompt = prompts.EBM_rewriting_dict[EBM_type] if EBM_type in prompts.EBM_rewriting_dict else prompts.EBM_rewriting_dict["others"] # still in dict format
        if language == "zh": # Chinese
            prompt = "你是一个查询语句的改写专家，" + prompt["zh"] + "直接给出改写后的查询语句，不要有其他表述。原始查询语句为:\n\n" + raw_text
        else: # English
            prompt = "You are a query sentence rewriting expert, " + prompt["en"] + "Directly provide the rewritten query sentence without any additional statements. Original query sentence:\n\n" + raw_text
    
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # generate outputs
        outputs = self.llm.generate([text], self.sampling_params)
        response = outputs[0].outputs[0].text
        return response
    
                
    def handle_file(self, input_file_path, output_file_path, language="en"):
        # get line number
        line_num = subprocess.run(['wc', '-l', input_file_path], stdout=subprocess.PIPE)  
        line_count = int(line_num.stdout.split()[0])
        with open(input_file_path, "r", encoding="utf-8", errors="ignore") as fin, open(
                  output_file_path, "a", encoding="utf-8", errors="ignore") as fout:
            # for idx, line in enumerate(fin):
            for idx, line in enumerate(tqdm(fin, total=line_count)):
                try:
                    content = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
                    
                    # category
                    EBM_category = self.prompted_infer_category(content, "EBM_classification", language=language)
                    nlp_category = self.prompted_infer_category(content, "question_classification", language=language)
                    content["EBM_category"] = EBM_category
                    content["nlp_category"] = nlp_category
                    
                    # rewritten query
                    rewritten_query = self.query_rewriting(content, EBM_category, language=language)
                    content["rewritten_query"] = rewritten_query
                    
                    fout.write(ujson.dumps(content, ensure_ascii = False) + "\n")
                    
                except ValueError as e:
                    print(f"JSON parser error: {e}")
        

if __name__ == "__main__":
    output_root_folder = "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query_rewriting/output/"
    input_path_dict = {
        "MedQA": [
            # "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/SFT_series/benchmark/medical/MedQA/data_clean/questions/Mainland/chinese_qbank.jsonl", # MedQA
            # "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/SFT_series/benchmark/medical/MedQA/data_clean/questions/US/US_qbank.jsonl",
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/benchmark/MedQA/data_clean/questions/Mainland/dev.jsonl",
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/benchmark/MedQA/data_clean/questions/Mainland/test.jsonl",
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/benchmark/MedQA/data_clean/questions/US/dev.jsonl",
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/benchmark/MedQA/data_clean/questions/US/test.jsonl"
        ],
        # "MedMCQA": [
        #     "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/train.json",
        #     "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/dev.json",
        #     "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/test.json"
        # ]
    }
    
    model_path = "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/model/Qwen2.5-72B-Instruct/"
    qwen_infer = QwenInfer(model_path)
    
    for dataset, input_paths in input_path_dict.items():
        for input_path in input_paths:
            language_type = "en" # default
            output_folder = output_root_folder + dataset + "/"
            os.makedirs(output_folder, exist_ok=True)
            base_folder_name = input_path.split("/")[-2]
        #    print(f"Processing {base_folder_name}...")
        
            if base_folder_name in ["Mainland", "US"]:
                os.makedirs(output_folder + base_folder_name + "/", exist_ok=True)
                output_folder = output_folder + base_folder_name + "/" # update output_folder → subfolder
                if base_folder_name == "Mainland":
                    language_type = "zh"
            else: # medmcqa
                pass
            
            filename = os.path.basename(input_path)
                
            qwen_infer.handle_file(input_path, output_folder + filename, language=language_type) # (input_path, output_path)