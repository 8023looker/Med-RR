#!/bin/bash

# parallel -j 5 python ::: \
#             medicine_query_parallel.py MedQA /cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/MedQA/data_clean/questions/Mainland/chinese_qbank.jsonl \
#             medicine_query_parallel.py MedQA /cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/MedQA/data_clean/questions/US/US_qbank.jsonl \
#             medicine_query_parallel.py MedMCQA /cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/train.json \
#             medicine_query_parallel.py MedMCQA /cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/dev.json \
#             medicine_query_parallel.py MedMCQA /cpfs/29f69eb5e2e60f26/code/CPT_params/SFT_series/benchmark/medical/medmcqa/medmcqa_data/test.json

# parallel -j 5 -n 2 python medicine_query_parallel.py ::: \
#     MedQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/Mainland/chinese_qbank.jsonl \
#     MedQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/US/US_qbank.jsonl \
#     MedMCQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedMCQA/train.json \
#     MedMCQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedMCQA/dev.json \
#     MedMCQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedMCQA/test.json 

parallel -j 1 -n 2 python medicine_query_parallel_dev.py ::: \
    MedQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/benchmark/MedQA/data_clean/questions/Mainland/dev.jsonl
    # MedQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/benchmark/MedQA/data_clean/questions/Mainland/test.jsonl \
    # MedQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/benchmark/MedQA/data_clean/questions/US/dev.jsonl \
    # MedQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/benchmark/MedQA/data_clean/questions/US/test.jsonl

# parallel -j 4 -n 2 python medicine_query_parallel_dev.py ::: \
#     MedQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/Mainland/dev.jsonl \
#     MedQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/Mainland/test.jsonl \
#     MedQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/US/dev.jsonl \
#     MedQA /cpfs/29f69eb5e2e60f26/code/medical_RAG/RAG/query_rewriting/output/MedQA/US/test.jsonl
