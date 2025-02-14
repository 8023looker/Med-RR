" Qwen-72B CoT Generation "
""" 指导如何根据根据开卷资料进行分析并查找答案 (How to use the retrieved documents) """
import os
import json

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ujson
from tqdm import tqdm
import subprocess

import prompts

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


class QwenCoTGeneration:
    def __init__(self, model_path):
        self.llm = LLM(model=model_path, tensor_parallel_size=4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

     
    def cot_generate(self, prompt, language="en"):
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
    
                
    def handle_file(self, input_query_path, input_retrieval_folder, output_file_path, dataset_name, language="en", query_key="question"):
        with open(input_query_path, "r", encoding="utf-8", errors="ignore") as fin, open(
                  output_file_path, "a", encoding="utf-8", errors="ignore") as fout:
            for idx, line in enumerate(fin):
                try:
                    content = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
                    query_question = content[query_key] if dataset_name != "PubMedQA" else content[query_key.upper()] # "question" or "query_rewritten" or "QUESTION" 
                    
                    index = idx if dataset_name != "PubMedQA" else content["idx"]
                    
                    retrieval_list = []
                    with open(input_retrieval_folder + query_key + f"/result_{str(index)}.jsonl", "r", encoding="utf-8", errors="ignore") as f_retrieval: # 对于 MedQA, MedMCQA
                        for idx_retrieval, line_retrieval in enumerate(f_retrieval):
                            try:
                                retrieval_content = ujson.loads(line_retrieval.replace("\n", "").replace("\\/", "/"))
                                chunk = retrieval_content["fields"]["chunk"]
                                retrieval_list.append(chunk)
                            except ValueError as e:
                                print(f"JSON parser error: {e}")

                    retrieval_string = "\n\n".join(retrieval_list)
                    cot_generation_prompt = prompts.COT_PROMPTS[language] + "\n\n" + prompts.REPLACE_DICT["question"][language] + "\n\n" + query_question + "\n\n" + prompts.REPLACE_DICT["retrieved_documents"][language] + "\n\n" + retrieval_string
                    content["cot_generation"] = self.cot_generate(cot_generation_prompt, language)
                    fout.write(ujson.dumps(content, ensure_ascii = False) + "\n")
                           
                except ValueError as e:
                    print(f"JSON parser error: {e}")
        

def get_pubmedqa_index():
    pubmedqa_path = "/cpfs/29f69eb5e2e60f26/code/medical_RAG/benchmark/pubmedqa/data/ori_pqal.json"
    pubmedqa_idx_list = []
    with open(pubmedqa_path, "r", encoding="utf-8", errors="ignore") as fin:
        jdict = json.load(fin)


if __name__ == "__main__":
    retrieval_folder_dict = { # original retrieved documents
        "MedQA": [
            # "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedQA/Mainland/", # MedQA
            # "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedQA/US/", 
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedQA/Mainland/dev/",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedQA/Mainland/test/",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedQA/US/dev/",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedQA/US/test/"
        ],
        "MedMCQA": [
            # "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedMCQA/train/",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedMCQA/dev/",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedMCQA/test/"
        ],
        # "PubMedQA": [ # 待修改，缺少 index → 已修改
        #     "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/PubMedQA/ori_pqal/"
        # ]
    }
    query_folder_dict_rewritten = { # rewritten queries
        # "MedQA": [
        #     "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/Mainland/chinese_qbank.jsonl", # MedQA
        #     "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/US/US_qbank.jsonl",  
        # ],
        "MedMCQA": [
            # "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedMCQA/train.json",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedMCQA/dev.json",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedMCQA/test.json"
        ],
        # "PubMedQA": [
        #     "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/PubMedQA/ori_pqal.json"
        # ]
    }
    
    query_folder_dict = { # rewritten
        "MedQA": [
            # "/cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/MedQA/data_clean/questions/Mainland/chinese_qbank.jsonl", # MedQA
            # "/cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/MedQA/data_clean/questions/US/US_qbank.jsonl", 
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/Mainland/dev.jsonl",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/Mainland/test.jsonl",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/US/dev.jsonl",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/US/test.jsonl",
        ],
        "MedMCQA": [
            # "/cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/train.json",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedMCQA/dev.json",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedMCQA/test.json"
        ],
        # "PubMedQA": [ # rewritten
        #     "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/PubMedQA/ori_pqal.json"
        # ]
    }
    
    output_root_folder = "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/CoT/output/"
    
    model_path = "/cpfs/29f69eb5e2e60f26/code/model/Qwen2.5-72B-Instruct/"
    qwen_cot_generate = QwenCoTGeneration(model_path)
    
    language = "en" # default
    for dataset, input_paths in query_folder_dict.items():
        for idx, input_query_path in enumerate(input_paths):
            input_retrieval_folder = retrieval_folder_dict[dataset][idx]
            output_folder = output_root_folder + dataset + "/"
            os.makedirs(output_folder, exist_ok=True) # MedQA, MedMCQA, PubMedQA folder
            
            base_folder_name = input_query_path.split("/")[-2]
            if base_folder_name in ["Mainland", "US"]: # medqa
                os.makedirs(output_folder + base_folder_name + "/", exist_ok=True)
                output_folder = output_folder + base_folder_name + "/" # update output_folder → subfolder
                if base_folder_name == "Mainland":
                    language_type = "zh"
            else: # medmcqa or pubmedqa
                pass
            
            filename = os.path.basename(input_query_path) # filename = "train.json"
            pure_filename = filename.split(".")[0]
            output_file_path = output_folder + filename.split(".")[0] + "_written." + filename.split(".")[-1]
            
            # "question" or "query_rewritten" or "QUESTION"
            qwen_cot_generate.handle_file(input_query_path, input_retrieval_folder, output_file_path, dataset, language, query_key="rewritten_query") # (input_query_path, input_retrieval_folder, output_file_path, dataset_name, query_key=["question", "QUESTION", "rewritten_query"])
