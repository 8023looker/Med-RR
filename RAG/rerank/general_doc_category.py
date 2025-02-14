""" 基于 retrieval 召回的语句类型分类 """
""" General Document Category """

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
                "argumentation": "argumentation",
                "definition": "definition",
                "description": "description",
                "explanation": "explanation",
                "purpose": "purpose",
                "narration": "narration",
                "process": "process",
                "instruction": "instruction",
                "command": "command",
                "problem-solving": "problem-solving",
                "comparison": "comparison",
                "evaluation": "evaluation",
                "classification": "classification",
                "condition": "condition",
                "prediction": "prediction",
                "cause-and-effect": "cause-and-effect"
            },
            "zh": {
                "论证": "argumentation",
                "定义": "definition",
                "描述": "description",
                "解释": "explanation",
                "目的": "purpose",
                "叙述": "narration",
                "过程": "process",
                "指令": "instruction",
                "命令": "command",
                "问题解决": "problem-solving",
                "比较": "comparison",
                "评价": "evaluation",
                "分类": "classification",
                "条件": "condition",
                "预测": "prediction",
                "因果": "cause-and-effect"
            }
        }
        
        if topic not in category_trans_dict[language]:
            return f"error-{topic}"
        else:
            return category_trans_dict[language][topic]
    
    
    def general_document_category_score(self, distribution_list, nlp_category):
        document_classification_list = prompts.document_classification_list
        expected_category_list = self.expected_category(nlp_category) # list of expected categories
        sum_score = 0.0
        for idx, category in enumerate(document_classification_list):
            if category.lower() in expected_category_list:
                sum_score += distribution_list[idx]
        return sum_score
     
    def retrieval_infer_category(self, retrieval_chunk, language="en"): # 传入的是已经解析好的 jsonl_line_content 
        prompt = prompts.retrieval_category_prompt[language] + "\n\n" + retrieval_chunk
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
        return list(response)
    
    
    def expected_category(self, nlp_category):
        # nlp question-answer mapping
        retrieval_category = []
        if nlp_category in ["factual", "referential", "definition", "explanatory", "descriptive"]:
            retrieval_category = prompts.query_document_projection["factual"]
        elif nlp_category in ["directive", "opinion", "procedural"]:
            retrieval_category = prompts.query_document_projection["procedural"]
        elif nlp_category in ["comparative", "evaluative", "verification"]:
            retrieval_category = prompts.query_document_projection["comparative"]
        elif nlp_category in ["hypothetical"]:
            retrieval_category = prompts.query_document_projection["hypothetical"]
        
        return retrieval_category
    
                
    def handle_file(self, input_query_path, input_retrieval_folder, output_folder, dataset_name, language="en", query_key="question"):
        output_folder = output_folder + query_key + "/"
        os.makedirs(output_folder, exist_ok=True)
        
        with open(input_query_path, "r", encoding="utf-8", errors="ignore") as fin:
            for idx, line in enumerate(fin):
                try:
                    content = ujson.loads(line.replace("\n", "").replace("\\/", "/"))

                    nlp_category = content["nlp_category"]
                    
                    # if dataset_name == "PubMedQA": # different format
                    #     pass
                    
                    index = idx if dataset_name != "PubMedQA" else content["idx"]
                    
                    retrieval_list = []
                    with open(input_retrieval_folder + query_key + f"/result_{str(index)}.jsonl", "r", encoding="utf-8", errors="ignore") as f_retrieval: # 对于 MedQA, MedMCQA
                        for idx_retrieval, line_retrieval in enumerate(f_retrieval):
                            try:
                                retrieval_content = ujson.loads(line_retrieval.replace("\n", "").replace("\\/", "/"))
                                
                                chunk = retrieval_content["fields"]["chunk"]
                                chunk_category_distribution = self.retrieval_infer_category(chunk, language)
                                category_score = self.general_document_category_score(chunk_category_distribution, nlp_category)
                                retrieval_content["category_score"] = category_score # general document category score
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
