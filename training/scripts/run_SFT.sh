DATASET_LIST=(
            # /data_train2/code/data/keerlu/medical_RAG/benchmark/pubmedqa/data/train_set.json)
            #   /global_data/data/keerlu/medical_RAG/benchmark/MedQA/data_clean/questions/US/train.jsonl
            #   /global_data/data/keerlu/medical_RAG/benchmark/MedQA/data_clean/questions/Mainland/train.jsonl
              /global_data/data/keerlu/medical_RAG/benchmark/medmcqa/data/train.json)

MODEL_LIST=(
            # /train_data_load/keerlu_model/keerlu_model/Qwen2.5-14B/ 
            # /train_data_load/keerlu_model/keerlu_model/llama2-13b-hf/
            /global_data/data/opensource/qwen2d5-32b-instruct/)

for MODEL_DIR in "${MODEL_LIST[@]}"; do
    for DATA_DIR in "${DATASET_LIST[@]}"; do
        echo $(basename $MODEL_DIR)
        OUTPUT_DIR="../output_32b/"
        if [[ "$DATA_DIR" == *"US"* ]]; then
            echo "当前路径包含 'US'"
            OUTPUT_DIR="../output_32b/$(basename $MODEL_DIR)_MedQA_US"
        elif [[ "$DATA_DIR" == *"Mainland"* ]]; then
            echo "当前路径包含 'Mainland'"
            OUTPUT_DIR="../output_32b/$(basename $MODEL_DIR)_MedQA_Mainland"
        elif [[ "$DATA_DIR" == *"medmcqa"* ]]; then
            echo "当前路径包含 'MedMCQA'"
            OUTPUT_DIR="../output_32b/$(basename $MODEL_DIR)_MedMCQA"
        elif [[ "$DATA_DIR" == *"pubmedqa"* ]]; then
            echo "当前路径包含 'PubMedQA'"
            OUTPUT_DIR="../output_32b/$(basename $MODEL_DIR)_PubMedQA"
        else
            echo "Unknown dataset name: $DATA_DIR. Please specify a valid dataset directory."
        fi

        DECODER_LAYER="LlamaDecoderLayer"
        if [[ "$MODEL_DIR" == *"llama"* ]]; then
            echo "Using LlamaDecoderLayer for model: $MODEL_DIR"
            DECODER_LAYER="LlamaDecoderLayer"
        elif [[ "$MODEL_DIR" == *"qwen"* ]]; then
            echo "Using Qwen2DecoderLayer for model: $MODEL_DIR"
            DECODER_LAYER="Qwen2DecoderLayer"
        elif [[ "$MODEL_DIR" == *"Mistral"* ]]; then
            echo "Using MistralDecoderLayer for model: $MODEL_DIR"
            DECODER_LAYER="MistralDecoderLayer"
        else
            echo "Unknown model type: $MODEL_DIR. Please specify a valid model directory."
            exit 1
        fi

        # CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 \
        # --bf16 True \ # H20 注释掉
        # --num_train_epochs 3 \
        torchrun --nproc_per_node=8 --master_port=4585 ../src/train.py \
            --model_name_or_path ${MODEL_DIR} \
            --data_path ${DATA_DIR} \
            --output_dir ${OUTPUT_DIR} \
            --bf16 True \
            --num_train_epochs 3 \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --eval_strategy "no" \
            --save_strategy "steps" \
            --save_steps 2000 \
            --save_total_limit 1 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --fsdp "full_shard auto_wrap offload" \
            --fsdp_transformer_layer_cls_to_wrap "${DECODER_LAYER}" \
            --tf32 True \
            --overwrite_output_dir True \
            --gradient_checkpointing True \
            # --seed 42 # random seed

    done

done

# 13b
# --per_device_train_batch_size 2 \
# --per_device_eval_batch_size 4 \
# --gradient_accumulation_steps 8 \

# 7b
# --per_device_train_batch_size 8 \
# --per_device_eval_batch_size 8 \
# --gradient_accumulation_steps 16 \