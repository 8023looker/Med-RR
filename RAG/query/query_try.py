# test for search API
import ujson
import json
import requests

class MedicalQuery:
    def __init__(self, hit_size=20, timeout=2000):
        self.url = "http://medical-search.v1.test.c8de7188aa7bd4fdaa4d4e33f5f708da0.cn-beijing.alicontainer.com/v1/document/search"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'xxxxxx',
            # 'traceparent': traceparent  # 替换为实际的 traceparent
        }

    def query(self, query_data):
        response = requests.post(self.url, headers=self.headers, data=json.dumps(query_data))
        # 检查响应状态码
        if response.status_code == 200:
            print("Request successful!")
            print("Response:", response.json())  # 假设响应是 JSON 格式
        else:
            print(f"Request failed with status code: {response.status_code}")
            print("Response:", response.text)

# # 定义请求体
# data = {
#     "request_id": "keerlutest",
#     "query": "high stress lack of rest health prevention lifestyle advice",
#     "control": {
#         "hit_size": 20,
#         "timeout": 1000,
#     # "debug_level": 1
#     }
# }

# # POST
# response = requests.post(url, headers=headers, data=json.dumps(data))

# # 检查响应状态码
# if response.status_code == 200:
#     print("Request successful!")
#     print("Response:", response.json())  # 假设响应是 JSON 格式
# else:
#     print(f"Request failed with status code: {response.status_code}")
#     print("Response:", response.text)


if __name__ == "__main__":
    medical_query = MedicalQuery()
    query_data_example = {
        "request_id": "keerlutest",
        "query": "high stress lack of rest health prevention lifestyle advice",
        "control": {
            "hit_size": 20,
            "timeout": 1000,
        # "debug_level": 1
        }
    }
    
    query_data = {
        "request_id": "keerlutest",
        "query": "Studies on the long-term prognosis of breast cancer patients, focusing on 5-year survival rates.",
        "control": {
            "hit_size": 20,
            "timeout": 1000,
        # "debug_level": 1
        }
    }
    medical_query.query(query_data)