# step 1: embedding
import ujson
import json
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
import torch  
import torch.nn as nn
import os
import io
import sys
from tqdm import tqdm

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode, encoding='utf-8', errors='ignore')
    return f

class ModelBGEM3:
    def __init__(self, model_path='/global_data/data/bge/models/bge-m3'):
        self.model = BGEM3FlagModel(model_path,  
                       use_fp16=True)
        if torch.cuda.device_count() > 1:
            print("There are", torch.cuda.device_count(), "GPUs~")
            self.model = nn.DataParallel(self.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
    def get_paragraphs(self, file_content): # content → a single file
        # store the result_dict, text
        text_list, raw_text_list = [], []
        for line in file_content: # 1st round
            try:
                raw_text_list.append(line["fields"]["chunk"])
            except:
                print("Error: ", line)
    
        # use bge_m3 model to embed
        embedding_list = self.model.module.encode(raw_text_list, 
                                                  batch_size=3, 
                                                  max_length=8192,
                                            )['dense_vecs']
        # store the results
        emb_index = 0
        for line in file_content:
            line["vector_encoded"] = embedding_list[emb_index].tolist()
            text_list.append(line)
            emb_index += 1
        return text_list
                
    def handle_file(self, args): # args = [input_file, output_file]
        doc = []
        f = _make_r_io_base(args[0], "r")
        for idx, line in enumerate(f): # for medqa
            doc.append(json.loads(line))
        # with open(args[0], 'r') as file:  
        #     doc = ujson.load(file)
            
        rs_list = self.get_paragraphs(doc)
        with open(args[1], "w", encoding="utf-8") as fout:
            # ujson.dump(rs_list, fout, ensure_ascii=False)
            for idx, rs in enumerate(rs_list):
                fout.write(ujson.dumps(rs, ensure_ascii=False) + '\n')
            

if __name__ == "__main__":
    output_root_folder = "/global_data/data/medical_RAG/RAG/query/round_ablation/output/retrieval/"
    # output_root_folder = sys.argv[1] + "/"
    os.makedirs(output_root_folder, exist_ok=True)
    
    input_path_list = [
        "/global_data/data/medical_RAG/RAG/query/output/MedMCQA/dev/question/result_8.jsonl", 
        "/global_data/data/medical_RAG/RAG/query/output/MedMCQA/dev/rewritten_query/result_8.jsonl"
    ]
    
    embedding_pipeline = ModelBGEM3()
    
    for input_path in input_path_list:
        subfolder_name = input_path.split("/")[-1].split(".")[0]
        output_folder = output_root_folder + subfolder_name + "/"
        os.makedirs(output_folder, exist_ok=True)
        embedding_pipeline.handle_file([input_path, output_folder + input_path.split("/")[-2] + ".jsonl"])
    
