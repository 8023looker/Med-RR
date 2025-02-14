""" Hierarchy of Evidence """

import os
import ujson
import json

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ujson
from tqdm import tqdm
import subprocess

import prompts

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

class QwenRerank:
    def __init__(self, model_path):
        self.llm = LLM(model=model_path, tensor_parallel_size=4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
        
       
    def translate_category(self, topic, language): # topic: "描述" → "description"
        topic = topic.lower() # lowercase
        
        category_trans_dict = {
            "en": {
                "meta-analyses": "meta-analyses", 
                "systematic reviews": "systematic reviews", 
                "evidence-based practice guidelines": "evidence-based practice guidelines", 
                "randomized controlled trials": "randomized controlled trials", 
                "non-randomized controlled trials": "non-randomized controlled trials", 
                "cohort studies": "cohort studies", 
                "case series or studies": "case series or studies",
                "individual case reports": "individual case reports", 
                "expert opinion": "expert opinion"
            },
            "zh": {
                "元分析": "meta-analyses", 
                "系统综述": "systematic reviews", 
                "循证实践指南": "evidence-based practice guidelines", 
                "随机对照试验": "randomized controlled trials", 
                "非随机对照试验": "non-randomized controlled trials", 
                "队列研究": "cohort studies", 
                "病例系列或研究": "case series or studies",
                "个案报告": "individual case reports", 
                "专家意见": "expert opinion"
            }
        }
        
        if topic not in category_trans_dict[language]:
            return f"error-{topic}"
        else:
            return category_trans_dict[language][topic]
    
    
    def evidence_level_score(self, doc_level):
        evidence_hierarchy_score = prompts.evidence_level[doc_level]
        return evidence_hierarchy_score
    
    
    def retrieval_infer_category(self, retrieval_chunk, language="en"): # 传入的是已经解析好的 jsonl_line_content 
        prompt = prompts.evidence_level_prompt[language] + "\n\n" + retrieval_chunk
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
        response = outputs[0].outputs[0].text # [x1, x2, x3, ..., x16]
        # response_translate = self.translate_category(response, language) # original version: infer single category
        # print(response_translate)
        return response
    
                
    def handle_file(self, input_query_path, input_retrieval_folder, output_folder, dataset_name, language="en", query_key="question"):
        output_folder = output_folder + query_key + "/"
        os.makedirs(output_folder, exist_ok=True)
        
        with open(input_query_path, "r", encoding="utf-8", errors="ignore") as fin:
            for idx, line in enumerate(fin):
                try:
                    content = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
 
                    # if dataset_name == "PubMedQA": # different format
                    #     pass
                    
                    index = idx if dataset_name != "PubMedQA" else content["idx"]
                    
                    retrieval_list = []
                    with open(input_retrieval_folder + query_key + f"/result_{str(index)}.jsonl", "r", encoding="utf-8", errors="ignore") as f_retrieval: # 对于 MedQA, MedMCQA
                        for idx_retrieval, line_retrieval in enumerate(f_retrieval):
                            try:
                                retrieval_content = ujson.loads(line_retrieval.replace("\n", "").replace("\\/", "/"))

                                chunk = retrieval_content["fields"]["chunk"]
                                chunk_level = self.retrieval_infer_category(chunk, language)
                                if retrieval_content["collection"] == "medical-guideline": # guideline collection
                                    chunk_level = "evidence-based practice guidelines"
                                level_score = self.evidence_level_score(chunk_level)
                                retrieval_content["evidence_level_score"] = level_score # hierarchy of evidence
                                retrieval_list.append(retrieval_content)
                                
                            except ValueError as e:
                                print(f"JSON parser error: {e}")

                    with open(output_folder + f"result_{str(index)}.jsonl", "a", encoding="utf-8", errors="ignore") as fout:
                        for result in retrieval_list:
                            fout.write(ujson.dumps(result, ensure_ascii = False) + "\n")
                           
                except ValueError as e:
                    print(f"JSON parser error: {e}")


if __name__ == "__main__":
    retrieval_folder_dict = {
        "MedQA": [
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedQA/Mainland/", # MedQA
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedQA/US/",        
        ],
        "MedMCQA": [
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedMCQA/train/",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedMCQA/dev/",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/MedMCQA/test/"
        ],
        "PubMedQA": [
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/PubMedQA/ori_pqal/"
        ]
    }
    query_folder_dict = {
        "MedQA": [
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/Mainland/chinese_qbank.jsonl", # MedQA
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/US/US_qbank.jsonl",  
        ],
        "MedMCQA": [
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedMCQA/train.json",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedMCQA/dev.json",
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedMCQA/test.json"
        ],
        "PubMedQA": [
            "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/PubMedQA/ori_pqal.json"
        ]
    }
    
    output_root_folder = "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/rerank/output/"
    
    model_path = "/cpfs/29f69eb5e2e60f26/code/model/Qwen2.5-72B-Instruct/"
    qwen_rerank = QwenRerank(model_path)
    
    language = "en" # default
    for dataset, input_paths in query_folder_dict.items():
        for idx, input_query_path in enumerate(input_paths):
            input_retrieval_folder = retrieval_folder_dict[dataset][idx]
            output_folder = output_root_folder + dataset + "/"
            os.makedirs(output_folder, exist_ok=True) # MedQA, MedMCQA, PubMedQA folder
            
            filename = os.path.basename(input_query_path) # filename = "train.json"
            os.makedirs(output_folder + filename.split(".")[0] + "/", exist_ok=True)
            output_folder = output_folder + filename.split(".")[0] + "/" # update output_folder → subfolder
            
            base_folder_name = input_query_path.split("/")[-2]
            if base_folder_name == "Mainland":
                language = "zh"
                
            qwen_rerank.handle_file(input_query_path, input_retrieval_folder, output_folder, dataset, language, query_key="question") # (input_query_path, input_retrieval_folder, output_folder, dataset_name, query_key=["question", "QUESTION", "query_rewritten"])
            # qwen_rerank.handle_file(input_query_path, input_retrieval_folder, output_folder, dataset, language, query_key="query_rewritten") # (input_query_path, input_retrieval_folder, output_folder, dataset_name, query_key=["question", "QUESTION", "query_rewritten"])
