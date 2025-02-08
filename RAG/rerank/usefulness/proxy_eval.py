#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
os.environ["WANDB_MODE"] = "offline" # emmm 服务器连不上 wandb...

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

import utils
from loss_based_func import *

import ujson
import json

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": ( # 拥有Input的情况
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": ( # 没有Input的情况
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_medqa": (
        "Below is a single-choice question-answering task related to the medical field. "
        "Consult the retrieved documents, identify key information that are directly related to the question, perform your thought process based on the retrieved information and your own knowledge step by step in your mind, then construct and output your answer.\n\n"
        "### Retrieved Documents:\n{documents}\n\n"
        "### Example Thought Process:\n{cot_generation}\n\n" # cot_generation
        "Simply provide the choices of the options, without any other expressions.\n\n"
        # "### Example:\n### Question:Which of the following is not true for myelinated nerve fibers:\n\n### Options:\nA:Impulse through myelinated fibers is slower than non-myelinated fibers\nB:Membrane currents are generated at nodes of Ranvier\nC:Saltatory conduction of impulses is seen\nD:Local anesthesia is effective only when the nerve is not covered by myelin sheath\n\n### Answer:A\n\n"
        "### Question:\n{question}\n\n### Options:\n{options_str}\n### Answer:"
    ),
    "prompt_medmcqa": (
        "Below is a {choice_type}-choice question-answering task related to the medical field. "
        "Consult the retrieved documents, identify key information that are directly related to the question, perform your thought process based on the retrieved information and your own knowledge step by step in your mind, then construct and output your answer.\n\n"
        "### Retrieved Documents:\n{documents}\n\n"
        "### Example Thought Process:\n{cot_generation}\n\n"
        "Simply provide the choices of the options, without any other expressions.\n\n"
        # "### Example:\n### Question:Which of the following is not true for myelinated nerve fibers:\n\n### Options:\nA:Impulse through myelinated fibers is slower than non-myelinated fibers\nB:Membrane currents are generated at nodes of Ranvier\nC:Saltatory conduction of impulses is seen\nD:Local anesthesia is effective only when the nerve is not covered by myelin sheath\n\n### Answer:A\n\n"
        "### Question:\n{question}\n\n### Options:\nA:{opa}\nB:{opb}\nC:{opc}\nD:{opd}\n\n### Answer:"
    ),
    "prompt_pubmedqa": (
        "Below is a question-answering task related to the medical field. "
        "Consult the retrieved documents, identify key information that are directly related to the question, perform your thought process based on the retrieved information and your own knowledge step by step in your mind, then construct and output your answer.\n\n"
        "### Retrieved Documents:\n{documents}\n\n"
        "### Example Thought Process:\n{cot_generation}\n\n"
        "Simply provide the final decision, without any other expressions.\n\n"
        "### Context:\n{CONTEXTS}\n\n### Question:\n{QUESTION}\n\n### Answer:"
        # "### Example:\n### Question:Which of the following is not true for myelinated nerve fibers:\n\n### Options:\nA:Impulse through myelinated fibers is slower than non-myelinated fibers\nB:Membrane currents are generated at nodes of Ranvier\nC:Saltatory conduction of impulses is seen\nD:Local anesthesia is effective only when the nerve is not covered by myelin sheath\n\n### Answer:A\n\n"
        # "### Question:\n{question}\n\n### Options:\nA:{opa}\nB:{opb}\nC:{opc}\nD:{opd}\n\n### Answer:"
    ),
    "prompt_mmlu_med": ( # for MMLU_med
        "Below is a single-choice question-answering task related to the medical field, paired with an example that provides further context. "
        "Consult the retrieved documents, identify key information that are directly related to the question, perform your thought process based on the retrieved information and your own knowledge step by step in your mind, then construct and output your answer.\n\n"
        "### Retrieved Documents:\n{documents}\n\n"
        "### Example Thought Process:\n{cot_generation}\n\n"
        "Simply provide the final decision, without any other expressions.\n\n"
        # "### Example:\n### Question:Which of the following is not true for myelinated nerve fibers:\n\n### Options:\nA:Impulse through myelinated fibers is slower than non-myelinated fibers\nB:Membrane currents are generated at nodes of Ranvier\nC:Saltatory conduction of impulses is seen\nD:Local anesthesia is effective only when the nerve is not covered by myelin sheath\n\n### Answer:A\n\n"
        "### Question:\n{Question}\n\n### Options:\nA:{opa}\nB:{opb}\nC:{opc}\nD:{opd}\n\n### Answer:"
    ),
    "prompt_medqa_ori": (
        "Below is a single-choice question-answering task related to the medical field, paired with an example that provides further context. "
        "Simply provide the choices of the options, without any other expressions.\n\n"
        # "### Example:\n### Question:Which of the following is not true for myelinated nerve fibers:\n\n### Options:\nA:Impulse through myelinated fibers is slower than non-myelinated fibers\nB:Membrane currents are generated at nodes of Ranvier\nC:Saltatory conduction of impulses is seen\nD:Local anesthesia is effective only when the nerve is not covered by myelin sheath\n\n### Answer:A\n\n"
        "### Question:\n{question}\n\n### Options:\n{options_str}\n### Answer:"
    ),
    "prompt_medmcqa_ori": (
        "Below is a {choice_type}-choice question-answering task related to the medical field, paired with an example that provides further context. "
        "Simply provide the choices of the options, without any other expressions.\n\n"
        # "### Example:\n### Question:Which of the following is not true for myelinated nerve fibers:\n\n### Options:\nA:Impulse through myelinated fibers is slower than non-myelinated fibers\nB:Membrane currents are generated at nodes of Ranvier\nC:Saltatory conduction of impulses is seen\nD:Local anesthesia is effective only when the nerve is not covered by myelin sheath\n\n### Answer:A\n\n"
        "### Question:\n{question}\n\n### Options:\nA:{opa}\nB:{opb}\nC:{opc}\nD:{opd}\n\n### Answer:"
    ),
    "prompt_pubmedqa_ori": (
        "Below is a question-answering task related to the medical field, paired with an example that provides further context. "
        "Simply provide the final decision, without any other expressions.\n\n"
        "### Context:\n{CONTEXTS}\n\n### Question:\n{QUESTION}\n\n### Answer:"
    ), 
    "prompt_mmlu_med_ori": ( # for MMLU_med
        "Below is a single-choice question-answering task related to the medical field, paired with an example that provides further context. "
        "Simply provide the choices of the options, without any other expressions.\n\n"
        "### Question:\n{Question}\n\n### Options:\nA:{opa}\nB:{opb}\nC:{opc}\nD:{opd}\n\n### Answer:"
    ),
}

DATA_QUERY_PROJ = {
    "/global_data/data/keerlu/medical_RAG/RAG/CoT/output/PubMedQA/ori_pqal.json": "/global_data/data/keerlu/medical_RAG/RAG/query/output/PubMedQA/ori_pqal/rewritten_query/", 
    "/global_data/data/keerlu/medical_RAG/RAG/CoT/output/MedQA/US/dev_written.jsonl": "/global_data/data/keerlu/medical_RAG/RAG/query/output/MedQA/US/dev/rewritten_query/", 
    "/global_data/data/keerlu/medical_RAG/RAG/CoT/output/MedQA/Mainland/dev_written.jsonl": "/global_data/data/keerlu/medical_RAG/RAG/query/output/MedQA/Mainland/dev/rewritten_query/", 
    "/global_data/data/keerlu/medical_RAG/RAG/CoT/output/MedMCQA/dev_written.json": "/global_data/data/keerlu/medical_RAG/RAG/query/output/MedMCQA/dev/rewritten_query/"
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class DataArguments_new:
    data_path: list[str] = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024, # 512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, choice: str):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        
        # judge dataset_name
        dataset_name = "MedMCQA" # default
        if "US" in data_path or "Mainland" in data_path:
            dataset_name = "MedQA"
        elif "PubMedQA" in data_path:
            dataset_name = "PubMedQA"
        elif "MedMCQA" in data_path:
            dataset_name = "MedMCQA"
        else:
            print("Error: dataset_name not found!")
        
        query_path = DATA_QUERY_PROJ[data_path]
        # list_data_dict = utils.jload_medical_cot(data_path, dataset_name, query_path) if choice == "rag" else utils.jload(data_path, dataset_name)
        list_data_dict = utils.jload(data_path, dataset_name) # key 中包含 "documents"
        logging.warning("Formatting inputs...")
        
        if choice == "ori":
            prompt_medqa, prompt_medmcqa, prompt_pubmedqa, prompt_mmlu_med = PROMPT_DICT["prompt_medqa_ori"], PROMPT_DICT["prompt_medmcqa_ori"], PROMPT_DICT["prompt_pubmedqa_ori"], PROMPT_DICT["prompt_mmlu_med_ori"]
        elif choice == "rag":
            prompt_medqa, prompt_medmcqa, prompt_pubmedqa, prompt_mmlu_med = PROMPT_DICT["prompt_medqa"], PROMPT_DICT["prompt_medmcqa"], PROMPT_DICT["prompt_pubmedqa"], PROMPT_DICT["prompt_mmlu_med"]

        sources = [
            prompt_medqa.format_map(example) if dataset_name == "MedQA" else (prompt_medmcqa.format_map(example) if dataset_name == "MedMCQA" else (prompt_pubmedqa.format_map(example) if dataset_name == "PubMedQA" else prompt_mmlu_med.format_map(example)))
            for example in list_data_dict
        ]
        
        if dataset_name == "MedMCQA":
            for example in list_data_dict:
                if example["cop"] == 1:
                    example["cop"] = "A"
                elif example["cop"] == 2:
                    example["cop"] = "B"
                elif example["cop"] == 3:
                    example["cop"] = "C"
                elif example["cop"] == 4:
                    example["cop"] = "D"
        targets = [f"{example['answer_idx']}{tokenizer.eos_token}" if dataset_name == 'MedQA' else 
                   (f"{example['cop']}{tokenizer.eos_token}" if dataset_name == 'MedMCQA' else
                   f"{example['final_decision']}{tokenizer.eos_token}") for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, choice) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, choice)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, choice)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


# dataloader processing
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
def get_dataloader(dataset, data_collator, batch_size=1): # batch_size = 1, 1 batch contains 1 data point
    dataloader = DataLoader(dataset,
                            batch_size=batch_size, # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader


def eval_loss_diff():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict, 
        tokenizer=tokenizer, 
        model=model, 
    )
    
    # retrieved documents folder
    output_root_folder = "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/rerank/usefulness/output/"
    # judge dataset_name
    dataset_name = "MedMCQA" # default
    if "US" in data_args.data_path or "Mainland" in data_args.data_path:
        dataset_name = "MedQA"
    elif "PubMedQA" in data_args.data_path:
        dataset_name = "PubMedQA"
    elif "MedMCQA" in data_args.data_path:
        dataset_name = "MedMCQA"
  
    input_retrieval_folder = "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query/output/" + dataset_name + "/"
    output_folder = output_root_folder + dataset_name + "/"
    os.makedirs(output_folder, exist_ok=True) # MedQA, MedMCQA, PubMedQA, mmlu_med folder
    
    filename = os.path.basename(data_args.data_path) # filename = "train.json"
    os.makedirs(output_folder + filename.split(".")[0] + "/", exist_ok=True)
    output_folder = output_folder + filename.split(".")[0] + "/" # update output_folder → subfolder
    base_folder_name = data_args.data_path.split("/")[-2]
    if base_folder_name == "Mainland":
        language = "zh"
    
    with open(data_args.data_path, "r", encoding="utf-8", errors="ignore") as fin:
        for idx, line in enumerate(fin):
            try:
                content = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
                with open(output_folder + f"result_{str(idx)}.jsonl", "a", encoding="utf-8", errors="ignore") as fout, open(
                          input_retrieval_folder + f"result_{str(idx)}.jsonl", "r", encoding="utf-8", errors="ignore") as f_retrieval, open(
                          input_retrieval_folder + f"result_{str(idx)}_new.jsonl", "a", encoding="utf-8", errors="ignore") as fout_retrieved_doc:
                    
                    for idx_retrieval, line_retrieval in enumerate(f_retrieval):
                        try:
                            retrieval_content = ujson.loads(line_retrieval.replace("\n", "").replace("\\/", "/"))
                            chunk = retrieval_content["fields"]["chunk"]
                            cur_content = copy.deepcopy(content)
                            cur_content["documents"] = chunk
                            fout.write(json.dumps(cur_content, ensure_ascii=False) + "\n")
                            
                        except ValueError as e:
                            print(f"JSON parser error: {e}")
                    
                    data_args_new = copy.deepcopy(data_args)
                    data_args_new.data_path = output_folder + f"result_{str(idx)}.jsonl"
                    
                    # load data
                    data_module_rag = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args_new, choice="rag")
                    eval_dataloader_rag = get_dataloader(data_module_rag["eval_dataset"], data_module_rag["data_collator"], batch_size=1)
                    
                    data_module_origin = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args_new, choice="ori")
                    eval_dataloader_origin = get_dataloader(data_module_rag["eval_dataset"], DataCollatorForSeq2Seq(tokenizer), batch_size=1)
                    
                    loss_trainer = LossTrainer(model=model, tokenizer=tokenizer, args=training_args)
                    loss_rag_list = loss_trainer.loss_based_evaluation(eval_dataloader_rag, model)
                    loss_origin_list = loss_trainer.loss_based_evaluation(eval_dataloader_origin, model)
                    
                    loss_diff_list = compute_loss_diff(loss_rag_list, loss_origin_list) # for current data point, loss_diff_list for each retrieved document
                    for idx_retrieval, line_retrieval in enumerate(f_retrieval):
                        try:
                            retrieval_content = ujson.loads(line_retrieval.replace("\n", "").replace("\\/", "/"))
                            retrieval_content["loss_diff"] = loss_diff_list[idx_retrieval]
                            fout_retrieved_doc.write(json.dumps(retrieval_content, ensure_ascii=False) + "\n")
                        except ValueError as e:
                            print(f"JSON parser error: {e}")
                    
            except ValueError as e:
                print(f"{e}")
    
    
def compute_loss_diff(loss_rag_list, loss_origin_list):
    loss_diff_list = []
    try:
        loss_diff_list = [max(b - a, 0) for a, b in zip(loss_rag_list, loss_origin_list)]
    except:
        print("Error: loss_rag_list and loss_origin_list have different lengths!")
    return loss_diff_list


if __name__ == "__main__":
    eval_loss_diff()