# 串行
import os
import io
import ujson
import json
import requests
from tqdm import tqdm

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode, encoding='utf-8', errors='ignore')
    return f

class MedicalQuery:
    def __init__(self, hit_size=20, timeout=2000):
        # 定义请求的 URL 和必要的头信息
        self.url = "http://medical-search.v1.test.c8de7188aa7bd4fdaa4d4e33f5f708da0.cn-beijing.alicontainer.com/v1/document/search"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'xxxxxx', # fill in
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

  
    def handle_file(self, input_path, output_folder, query_key="question"):
        output_folder = output_folder + query_key + "/"
        os.makedirs(output_folder, exist_ok=True)
        
        sub_folder_name = input_path.split("/")[-1].split(".")[0]
        output_folder = output_folder + sub_folder_name + "/"
        os.makedirs(output_folder, exist_ok=True)
        
        f = _make_r_io_base(input_path, "r")
        jdict = ujson.load(f)
        for value in jdict:
            try:
                content = value["data"]
                content["idx"] = value["id"]
                query_data = {
                    "request_id": "keerlutest",
                    "query": content["Question"] + "\n" + content["Correct Answer"],
                    "control": {
                        "hit_size": 20,
                        "timeout": 2000,
                    # "debug_level": 1
                    }
                }
                response = self.query(query_data, content["idx"])
                
                if "result" in response:
                    result_list = response["result"]["hits"] if "hits" in response["result"] else []
                    
                with open(output_folder + f"result_{str(content['idx'])}.jsonl", "a", encoding="utf-8", errors="ignore") as fout:
                    for result in result_list:
                        fout.write(ujson.dumps(result, ensure_ascii = False) + "\n")
            
            except ValueError as e:
                print(f"JSON parser error: {e}")
                    

if __name__ == "__main__":
    output_root_folder = "/global_data/data/medical_RAG/RAG/query/output/"
    # for raw data
    input_path_dict = {
        "MMLU_Med": []
    }
    
    medical_query = MedicalQuery() # initialize medical_query
    
    dataset_folder = "/global_data/data/medical_RAG/benchmark/mmlu_med/data/"
    for dirpath, dirnames, filenames in os.walk(dataset_folder):
        for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
            dataset_path = os.path.join(dirpath, filename)
            input_path_dict["MMLU_Med"].append(dataset_path)
    
    for dataset, input_paths in input_path_dict.items():
        for input_path in input_paths:
            output_folder = output_root_folder + dataset + "/"
            os.makedirs(output_folder, exist_ok=True)
            base_folder_name = input_path.split("/")[-2]
           
            # filename = os.path.basename(input_path)
            medical_query.handle_file(input_path, output_folder, query_key="question") # (input_path, output_folder)
