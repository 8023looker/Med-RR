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

  
    def handle_file(self, input_path, output_folder, dataset_name, query_key="question"):
        output_folder = output_folder + query_key + "/"
        os.makedirs(output_folder, exist_ok=True)
        
        with open(input_path, "r", encoding="utf-8", errors="ignore") as fin:
            for idx, line in enumerate(fin):
                try:
                    content = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
                    
                    if dataset_name == "MedQA":           
                        filtered_options = [option for option in content["options"] if isinstance(option, str)]
                        option_string = ', '.join(filtered_options)
                    elif dataset_name == "MedMCQA":
                        option_string = "\nOptions:\nA" + content["opa"] + "\nB" + content["opb"] + "\nC" + content["opc"] + "\nD" + content["opd"]
                        # question_string = content[query_key] + content["exp"] if content["exp"] is not null else content[query_key] # query_key = "question"
                    elif dataset_name == "PubMedQA":
                        contexts = '\n'.join(content["CONTEXTS"])
                        answer = content["final_decision"]
                    
                    query_data = {
                        "request_id": "keerlutest",
                        "query": content[query_key] + "\n" + option_string if dataset_name != "PubMedQA" else contexts + content[query_key] + "\n\nAnswer:" + answer,
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
                    
            # info: query done
            print(f"{input_path} done!")
        


if __name__ == "__main__":
    output_root_folder = "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query/output/"
    
    # dataset, input_path = sys.argv[1], sys.argv[2]
    parser = argparse.ArgumentParser(description="Process medical queries.")
    parser.add_argument('dataset_type', type=str, help='Type of the dataset (e.g., MedQA, MedMCQA)')
    parser.add_argument('file_path', type=str, help='Path to the input file')

    args = parser.parse_args()
    dataset, input_path = args.dataset_type, args.file_path
    
    # input_path_dict = {
    #     "MedQA": [
    #         "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/SFT_series/benchmark/medical/MedQA/data_clean/questions/Mainland/chinese_qbank.jsonl", # MedQA
    #         "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/SFT_series/benchmark/medical/MedQA/data_clean/questions/US/US_qbank.jsonl",
            
    #     ],
    #     "MedMCQA": [
    #         "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/train.json",
    #         "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/dev.json",
    #         "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/test.json"
    #     ]
    # }
    
    medical_query = MedicalQuery() # initialize medical_query
    
    language_type = "en" # default
    output_folder = output_root_folder + dataset + "/"
    os.makedirs(output_folder, exist_ok=True)
    base_folder_name = input_path.split("/")[-2]

    sub_folder_name = input_path.split("/")[-1].split(".")[0] # "train", "dev", "test", etc.

    if base_folder_name in ["Mainland", "US"]: # MedQA
        os.makedirs(output_folder + base_folder_name + "/", exist_ok=True)
        output_folder = output_folder + base_folder_name + "/" # update output_folder → subfolder
        os.makedirs(output_folder + sub_folder_name + "/", exist_ok=True)
        output_folder = output_folder + sub_folder_name + "/"
        if base_folder_name == "Mainland":
            language_type = "zh"
    elif base_folder_name in ["MedMCQA", "PubMedQA"]: # medmcqa, "PubMedQA"
        filename = os.path.basename(input_path) # filename = "train.json", "PubMedQA", etc.
        os.makedirs(output_folder + filename.split(".")[0] + "/", exist_ok=True)
        output_folder = output_folder + filename.split(".")[0] + "/" # update output_folder → subfolder
    
    # filename = os.path.basename(input_path)
    medical_query.handle_file(input_path, output_folder, dataset, query_key="question") # (input_path, output_folder)
    # medical_query.handle_file(input_path, output_folder, dataset, query_key="rewritten_query")