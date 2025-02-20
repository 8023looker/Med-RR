import os
import sys
import argparse
import ujson
import json
import requests
import transformers
    

class MedicalQuery:
    def __init__(self, hit_size=20, timeout=2000):
        # 定义请求的 URL 和必要的头信息
        self.url = "http://medical-search.v1.test.c8de7188aa7bd4fdaa4d4e33f5f708da0.cn-beijing.alicontainer.com/v1/document/search"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'xxxxx',
            # 'traceparent': traceparent  # 替换为实际的 traceparent
        }


    def query(self, query_data, idx):
        response = requests.post(self.url, headers=self.headers, data=json.dumps(query_data))
        # 检查响应状态码
        if response.status_code == 200:
            print(f"{idx} Request successful!")
            # print("Response:", response.json())  # 响应是 JSON 格式
            return response.json()
        else:
            print(f"{idx} Request failed with status code: {response.status_code}")
            # print("Response:", response.text)
            return response.text

  
    def handle_file(self, input_path, output_folder, dataset_name, query_key="question"):
        output_folder = output_folder + query_key + "/"
        os.makedirs(output_folder, exist_ok=True)
        
        with open(input_path, "r", encoding="utf-8", errors="ignore") as fin:
            jdict = json.load(fin)
            for key, value in jdict.items():
                if dataset_name == "PubMedQA":
                    try:
                        contexts = '\n'.join(value["CONTEXTS"])
                        question, answer = value["QUESTION"], value["final_decision"]

                        query_data = {
                            "request_id": "keerlutest",
                            "query": contexts + question + "\n\nAnswer:" + answer,
                            "control": {
                                "hit_size": 20,
                                "timeout": 2000,
                            # "debug_level": 1
                            }
                        }   
                        response = self.query(query_data, key)
                        
                        if "result" in response:
                            result_list = response["result"]["hits"] if "hits" in response["result"] else []
                            
                        with open(output_folder + f"result_{key}.jsonl", "a", encoding="utf-8", errors="ignore") as fout:
                            for result in result_list:
                                fout.write(ujson.dumps(result, ensure_ascii = False) + "\n")
                    
                    except ValueError as e:
                        print(f"JSON parser error: {e}")
    

if __name__ == "__main__":
    output_root_folder = "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query/output/"
    dataset, input_path = "PubMedQA", "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/benchmark/pubmedqa/data/ori_pqal.json"

    medical_query = MedicalQuery() # initialize medical_query
    
    language_type = "en" # default
    output_folder = output_root_folder + dataset + "/" # pubmedqa folder
    os.makedirs(output_folder, exist_ok=True)
    base_folder_name = input_path.split("/")[-2]
    
    filename = os.path.basename(input_path) # filename = "train.json"
    os.makedirs(output_folder + filename.split(".")[0] + "/", exist_ok=True)
    output_folder = output_folder + filename.split(".")[0] + "/" # update output_folder → subfolder
    
    # filename = os.path.basename(input_path)
    medical_query.handle_file(input_path, output_folder, dataset, query_key="question") # (input_path, output_folder)