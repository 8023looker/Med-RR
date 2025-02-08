import json
import math
import os
import random
from tqdm import tqdm

def combine_folder_data(input_file_list, output_path):
    output = {}
    for file in input_file_list:
        if file.endswith('.json'):
            with open(os.path.join(root_dir, file)) as f:
                data = json.load(f)
                output.update(data)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    
    
if __name__ == '__main__':
    root_dir = '/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/benchmark/pubmedqa/data/'
    train_file_list, dev_file_list = [], [] 
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
            file_path = os.path.join(dirpath, filename)
            base_folder_name = file_path.split("/")[-2]
            if "pqal" in base_folder_name:
                if "train" in filename:
                    train_file_list.append(file_path)
                elif "dev" in filename:
                    dev_file_list.append(file_path)
    print("Processing training data...")
    print(train_file_list)
    print(dev_file_list)
    output_folder = "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/benchmark/pubmedqa/data/"
    # combine_folder_data(train_file_list, output_folder + "train.jsonl")
    # combine_folder_data(dev_file_list, output_folder + "dev.jsonl")
    train_dev_list = train_file_list + dev_file_list
    combine_folder_data(train_dev_list, output_folder + "train_and_dev.jsonl")
    