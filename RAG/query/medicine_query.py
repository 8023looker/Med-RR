# 串行
import os
import ujson
import json
import requests


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
        with open(input_path, "r", encoding="utf-8", errors="ignore") as fin:
            for idx, line in enumerate(fin):
                try:
                    content = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
                    
                    option_string = ', '.join(str(content["options"]))
                    query_data = {
                        "request_id": "keerlutest",
                        "query": content[query_key] + "\n" + option_string,
                        "control": {
                            "hit_size": 20,
                            "timeout": 2000,
                        # "debug_level": 1
                        }
                    }
                    response = self.query(query_data, idx)
                    
                    if "result" in response:
                        result_list = response["result"]["hits"] if "hits" in response["result"] else []
                        
                    with open(output_folder + f"result_{str(idx)}.jsonl", "a", encoding="utf-8", errors="ignore") as fout:
                        for result in result_list:
                            fout.write(ujson.dumps(result, ensure_ascii = False) + "\n")
                
                except ValueError as e:
                    print(f"JSON parser error: {e}")
    


if __name__ == "__main__":
    output_root_folder = "/cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query/output/"
    # for raw data
    input_path_dict = {
        "MedQA": [
            "/cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/MedQA/data_clean/questions/Mainland/chinese_qbank.jsonl", # MedQA
            "/cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/MedQA/data_clean/questions/US/US_qbank.jsonl",
        ],
        "MedMCQA": [
            "/cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/train.json",
            "/cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/dev.json",
            "/cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/test.json"
        ]
    }
    
    medical_query = MedicalQuery() # initialize medical_query
    
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
            
            # filename = os.path.basename(input_path)
            medical_query.handle_file(input_path, output_folder, query_key="question") # (input_path, output_folder)
