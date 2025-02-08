import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
import ujson
import os.path as osp
from typing import Optional, Sequence, Union, Dict

import openai
import tqdm
from openai import openai_object
import copy
import random

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False


def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[Dict[str, str]], Dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        while True:
            try:
                shared_kwargs = dict(
                    model=model_name,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )
                completion_batch = openai.Completion.create(prompt=prompt_batch, **shared_kwargs)
                choices = completion_batch.choices

                for choice in choices:
                    choice["total_tokens"] = completion_batch.usage.total_tokens
                completions.extend(choices)
                break
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload_single(f, mode="r"): # for single data_dict
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def jload_proxy(f, mode="r"): # for proxy loss mode
    folder_name = f.split('/')[-3] # "law", "medical", "finance", "science", "other (dataset)"
    # print(f"Loading {folder_name}...")
    domain_name = folder_name if folder_name != "dataset" else "other"
    
    jdict = list()
    f = _make_r_io_base(f, mode)
    if domain_name == "science": # jsonl file
        # "conversation": [{"id" , "conversations": [{"from": , "value": }, {}]}, {}, ...]
        for line in f:
            cur_data_dict = json.loads(line)
            cur_data_dict["domain"] = domain_name # add "domain name"
            jdict.append(cur_data_dict) # {"query": , "response": }
        f.close()
    else: # json file
        # "finance", "law", "medical", "code", "general": [{"instruction" , "input" , "output" }, {}, ...]
        jdict = json.load(f)
        for idx, item in enumerate(jdict):
            jdict[idx]["domain"] = domain_name # add "domain name"
        f.close()
    
    jdict = [dict(item, domain=domain_name) for item in jdict]
    # for multi-turn conversation data, ensure the format is correct + reconstruct the data
    jdict = reconstruct_science_conversation_data(jdict) if domain_name == "science" else jdict

    return jdict

# 对于 shareGPT 的数据，确保其中的conversation的格式是正确的,也就是一条 huamn 一条 gpt   
def check_shareGPT_format(data_list):
    new_data_list = []  
    for idx, data_item in enumerate(data_list):
        data_flag = True
        human_request_list, gpt_response_list = [], []
        for chat_idx, chat_item in enumerate(data_item["conversations"]):  
            if chat_idx % 2 == 0:
                if chat_item["from"] != "human":  
                    print(f"Error: For index {idx}, expected 'human' but got '{chat_item['from']}'")
                    data_flag = False
                    break
                else:
                    human_request_list.append(chat_item["value"])
            else: # chat_idx % 2 == 1
                if chat_item["from"] != "gpt":  
                    print(f"Error: For index {idx}, expected 'gpt' but got '{chat_item['from']}'")
                    data_flag = False
                    break
                else:
                    gpt_response_list.append(chat_item["value"])
                
        if data_flag: # if 符合标准
            data_item.update({"human_request": human_request_list, "gpt_response": gpt_response_list}) # reconstruct the data
            new_data_list.append(data_item)
    
    data_list = new_data_list
    return data_list


def reconstruct_shareGPT_data_v1(data_list):
    for idx, data_item in enumerate(data_list):
        human_request_list, gpt_response_list = [], []
        for chat_idx, chat_item in enumerate(data_item["conversations"]):
            # reconstruct the data
            if chat_item["from"] == "gpt": # gpt response
                gpt_response_list.append({
                    "idx": chat_idx,
                    "value": chat_item["value"]
                })
            else: # human request
                human_request_list.append({
                    "idx": chat_idx,
                    "value": chat_item["value"]
                })

        data_item.update({"human_request": human_request_list, "gpt_response": gpt_response_list}) # reconstruct the data
        data_list[idx] = data_item
    return data_list


def reconstruct_shareGPT_data(data_list):
    new_data_list = []
    for idx, data_item in enumerate(data_list):
        human_request_list, gpt_response_list = [], []
        for chat_idx, chat_item in enumerate(data_item["conversations"]):
            if chat_idx == 0:
                if chat_item["from"] != "gpt":
                    human_request_list.append(chat_item["value"])
            else: # chat_idx = 1, 2, ...
                if chat_item["from"] == "gpt" and len(gpt_response_list) == len(human_request_list) - 1: # gpt response
                    gpt_response_list.append(chat_item["value"])
                if chat_item["from"] == "human" and len(gpt_response_list) == len(human_request_list):
                    human_request_list.append(chat_item["value"])
        if len(human_request_list) != len(gpt_response_list):
            print(f"Error: For index {idx}, the number of human requests {len(human_request_list)} and gpt responses {len(gpt_response_list)} are not equal.")
            human_request_list = human_request_list[:len(gpt_response_list)] # truncate the human requests
        
        if len(gpt_response_list) > 0 and len(human_request_list) > 0:
            data_item.update({"human_request": human_request_list, "gpt_response": gpt_response_list}) # reconstruct the data
            new_data_list.append(data_item)
        # data_list[idx] = data_item
    data_list = new_data_list
    return data_list

def reconstruct_science_conversation_data(data_list):
    new_data_list= []
    system_prompt = str()
    for idx, data_item in enumerate(data_list):
        human_request_list, gpt_response_list = [], []
        for chat_idx, chat_item in enumerate(data_item["conversation"]):
            if chat_idx == 0 and chat_item["from"] == "system":
                system_prompt = chat_item["value"]
            if chat_item["from"] != "human" and len(human_request_list) == len(gpt_response_list): # human request
                human_request_list.append(chat_item["value"])
            elif chat_item["from"] != "gpt" and len(gpt_response_list) == len(human_request_list) - 1:
                gpt_response_list.append(chat_item["value"])
        
        if len(human_request_list) > len(gpt_response_list):
            print(f"Error: For index {idx}, the number of human requests {len(human_request_list)} and gpt responses {len(gpt_response_list)} are not equal.")
            human_request_list = human_request_list[:len(gpt_response_list)] # truncate the human requests
        elif len(human_request_list) < len(gpt_response_list):
            print(f"Error: For index {idx}, the number of human requests {len(human_request_list)} and gpt responses {len(gpt_response_list)} are not equal.")
            gpt_response_list = gpt_response_list[:len(human_request_list)]
        
        if len(gpt_response_list) > 0 and len(human_request_list) > 0:
            data_item.update({"system_prompt": system_prompt, "human_request": human_request_list, "gpt_response": gpt_response_list}) # reconstruct the data
            new_data_list.append(data_item)
        # data_list[idx] = data_item
    data_list = new_data_list
    return data_list

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

def science_prompt_format_map(example): # for science conversations
    source_prompt_list = [] # for "sonnet3.5_science_conversations"
    for idx, human_request in enumerate(example["human_request"]):
        cur_prompt = example["system_prompt"] + f"### Instruction:\n{human_request}\n\n### Response:"
        # cur_prompt = example["system_prompt"] + f"### Instruction:\n{human_request}\n\n### Response:" if idx == 0 else f"### Instruction:\n{human_request}\n\n### Response:"
        source_prompt_list.append(cur_prompt)
    return source_prompt_list

def human_prompt_format_map(example): # for instruction following
    system_prompt_header = ( # initialize the prompt with the header
        "Below is an instruction that describes a scientific task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    # human_prompt_header = (
    #     "Below is a conversation about science between a curious human and an artificial intelligence assistant. "
    #     "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    #     "Provide a response that appropriately completes each request in this conversation.\n\n"
    # )
    source_prompt_list = [] # initialize the prompt with the header
    for idx, human_request in enumerate(example["human_request"]):
        cur_prompt = human_prompt_header + f"### Instruction:\n{human_request}\n\n### Response:" if idx == 0 else f"### Instruction:\n{human_request}\n\n### Response:"
        source_prompt_list.append(cur_prompt)
    return source_prompt_list

def human_target_format_map(example, tokenizer): # target
    target_prompt_list = []
    for gpt_response in example["gpt_response"]:
        cur_prompt = f"{gpt_response}{tokenizer.eos_token}"
        target_prompt_list.append(cur_prompt)
    return target_prompt_list
    
def concat_shareGPT_data(source_item, target_item): # source_item: human_request (list), target_item: gpt_response (list)
    combined_string = ""
    for idx, human_request in enumerate(source_item):
        combined_string += human_request + target_item[idx]
        if idx != len(source_item) - 1: # not the last one
            combined_string += "\n\n"
    return combined_string

def compute_conversation_source_input_len(source_token_list, target_token_list, tokenizer):
    source_offset = 0 # source, target 交替
    input_ids_lens_list = []
    for idx, s_token in enumerate(source_token_list): # [offset, length]
        source_length = s_token.input_ids.ne(tokenizer.pad_token_id).sum().item()
        target_length = target_token_list[idx].input_ids.ne(tokenizer.pad_token_id).sum().item()
        input_ids_lens_list.append((source_offset, source_length))
        source_offset += source_length + target_length # update the offset
    return input_ids_lens_list # [[offset0, length0], [offset1, length1], ...]


def compute_ratio(input_file_path, choice):
    topic_dict = {}
    with open(input_file_path, "r", encoding="utf-8", errors="ignore") as fin:
        content = ujson.load(fin) # dict format
        print(content)
        total_count = 0
        
        if choice == "versatune_dynamic":
            content = content["domain_weight"]
        
        for key in content:
            total_count += content[key]
            topic_dict[key] = content[key]
        for key in content:
            print(f"{key}: {content[key]} / {total_count} = {content[key] / total_count:.2f}")
            topic_dict[key] = content[key] / total_count
    return topic_dict


def initialize_model_params(input_file_path):
    initial_dict = {
        "domain_weight": {
            "law": 0.04,
            "medical": 0.09,
            "finance": 0.07,
            "science": 0.06,
            "code": 0.18,
            "other": 0.56
        },
        "model_path": "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/model/llama2-7b-hf/"
    }
    initial_dict["domain_weight"] = compute_ratio(input_file_path, choice="uniform")
    # with open("/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/HUAWEI/src/domain_weight.json", "w") as file:
    #     json.dump(initial_dict, file, indent=4)
    
    # file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/SFT_series/training/model_params.json"
    # with open(file_path, "r") as file:
    #     initial_dict = json.load(file)
    return initial_dict


def get_params(input_file_path):
    with open(input_file_path, "r", encoding="utf-8", errors="ignore") as file:
        params_dict = json.load(file)
    return params_dict

def write_params(output_file_path, params_dict):
    with open(output_file_path, "w", encoding="utf-8", errors="ignore") as file:
        json.dump(params_dict, file, ensure_ascii=False, indent=4)


# multi-ability domain weight
def initialize_multi_ability_domain_weight(choice, model_name, total_training_count=60000):
    domain_ratio_folder = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/SFT_series/domain_infer/result/"
    if choice == "versatune_dynamic":
        domain_ratio_folder = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/SFT_series/src/multi_ability/params/"
        # domain_ratio_folder = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/SFT_series/domain_infer/multi_ability_versatune_dynamic_result/"
    
    domain_weight_dict = compute_ratio(domain_ratio_folder + model_name + "_40k.json" if choice != "versatune_dynamic" else domain_ratio_folder + model_name + "_params.json", choice)
    
    if choice == "uniform":
        domain_weight_dict = {
            "law": 1/6,
            "medical": 1/6,
            "finance": 1/6,
            "science": 1/6,
            "code": 1/6,
            "other": 1/6
        }
    elif choice == "reverse_knowledge":
        for key in domain_weight_dict:
            domain_weight_dict[key] = 1 - domain_weight_dict[key]
        total_weight = sum(domain_weight_dict.values())
        for key in domain_weight_dict:
            domain_weight_dict[key] /= total_weight # renormalize
    elif choice == "versatune_constant" or choice == "versatune_dynamic":
        pass # 已经在 compute_ratio 中计算了
    
    return domain_weight_dict
    

def eval_domain_quanity():
    domain_count = { # ground_truth
        "law": 8241, # Lawyer Instruct
        "medical": 10178, # medical_meadow_medqa
        "finance": 68912, # finance alpaca
        "science": 8835, # sonnet3.5_science_conversations
        "code": 20022, # code Alpaca
        "other": 52002 # Alpaca 
    }
    return domain_count
    
def write_model_param(domain_weight_dict, model_path):
    file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/SFT_series/src/model_params.json"
    with open(file_path, "w", encoding="utf-8", errors="ignore") as fout:
        model_params_dict = {
            "domain_weight": domain_weight_dict,
            "model_path": model_path
        }
        print("model_params_dict", model_params_dict)
        json.dump(model_params_dict, fout, ensure_ascii=False, indent=4)
        
def read_model_param():
    file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/SFT_series/src/model_params.json"
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fin:
        model_params_dict = json.load(fin)
    return model_params_dict

def write2model_param(model_param_dict):
    file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/SFT_series/src/model_params.json"
    with open(file_path, "w", encoding="utf-8", errors="ignore") as fout:
        json.dump(model_param_dict, fout, ensure_ascii=False, indent=4)
        
def jload_pubmedqa(f, mode="r"): # for PubMedQA
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    jdict_list = []
    for key, value in jdict.items():
        value["idx"] = key
        value["CONTEXTS_STR"] = '\n'.join(value["CONTEXTS"])
        jdict_list.append(value)
    f.close()
    return jdict_list

def jload_mmlu_med(f, mode="r"): # for mmlu_med
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    jdict_list = []
    for value in jdict:
        content = value["data"]
        content["idx"] = value["id"]
        content["opa"] = content["Options"]["A"]
        content["opb"] = content["Options"]["B"]
        content["opc"] = content["Options"]["C"]
        content["opd"] = content["Options"]["D"]
        content["Correct_Answer"] = content["Correct Answer"]
        jdict_list.append(content)
    f.close()
    return jdict_list

def jload_medqa_jsonl(f, mode="r"): # for jsonl file
    """Load a .jsonl file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = []
    for line in f:
        cur_data_dict = json.loads(line)
        # flatten the options dict
        option_dict = cur_data_dict["options"]
        option_string = ""
        for key in option_dict:
            option_string = option_string + key + ": " + option_dict[key] + "\n"
        cur_data_dict["options_str"] = option_string
        jdict.append(cur_data_dict)
    f.close()
    return jdict

def jload_medmcqa_jsonl(f, mode="r"): # for jsonl file
    """Load a .jsonl file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = []
    for line in f:
        cur_data_dict = json.loads(line)
        jdict.append(cur_data_dict)
    f.close()
    return jdict

# 加载数据集
def jload(f, dataset_name):
    """Load a .json file into a dictionary."""
    if dataset_name == "PubMedQA":
        return jload_pubmedqa(f)
    elif dataset_name == "mmlu_med":
        return jload_mmlu_med(f)
    elif dataset_name == "MedQA":
        return jload_medqa_jsonl(f)
    elif dataset_name == "MedMCQA":
        return jload_medmcqa_jsonl(f)
    
def jload_medical_cot(f, dataset_name, query_folder, mode="r"): # for medical domain + CoT (MedQA_US, MedQA_Mainland, MedMCQA)
    """ Load a .json file into a dictionary. """
    """ Select 100 instances """
    
    sample_num = 1000 # if medmcqa, 1000; else: dev dataset
    sample_ratio = 0.01 if dataset_name == "MedMCQA" else 1.0
    
    f = _make_r_io_base(f, mode)
    jdict = []
    # value["idx"] = key

    for idx, line in enumerate(f): # for medqa
        if random.random() <= sample_ratio:
            cur_data_dict = json.loads(line)
            
            # retrieved documents
            query_file = query_folder + f"result_{str(idx)}.jsonl"
            # query_file = query_folder + f"result_{str(idx)}.jsonl" if dataset_name != "PubMedQA" else query_folder + f"result_{str(cur_data_dict['idx'])}.jsonl"
            doc_list = []
            with open(query_file, "r") as f_retrieval:
                for r_idx, line_retrieval in enumerate(f_retrieval):
                    retrieval_content = json.loads(line_retrieval.replace("\n", "").replace("\\/", "/"))
                    chunk = retrieval_content["fields"]["chunk"]
                    # doc_string += f"\n\nRetrieved Doc {str(r_idx + 1)}: " + chunk if r_idx != 0 else f"Retrieved Doc {str(r_idx + 1)}: " + chunk
                    doc_list.append(chunk)
            cur_data_dict["documents"] = doc_list
            
            if dataset_name == "MedQA":
                # flatten the options dict
                option_dict = cur_data_dict["options"]
                option_string = ""
                for key in option_dict:
                    option_string = option_string + key + ": " + option_dict[key] + "\n"
                cur_data_dict["options_str"] = option_string
            else: # MedMCQA
                pass
            
            if dataset_name == "MedQA" and idx == 3424:
                break
            
            jdict.append(cur_data_dict)
        
    f.close()
    return jdict

query_folder_dict = {
        "MedQA": [
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query/output/MedQA/Mainland/", # MedQA
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query/output/MedQA/US/",        
        ],
        "MedMCQA": [
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query/output/MedMCQA/train/",
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query/output/MedMCQA/dev/",
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query/output/MedMCQA/test/"
        ],
        "PubMedQA": [
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query/output/PubMedQA/ori_pqal/"
        ]
        "MMLU_med": [
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query_rewriting/output/mmlu_anatomy_test/", 
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query_rewriting/output/mmlu_clinical_knowledge_test/", 
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query_rewriting/output/mmlu_college_biology_test/", 
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query_rewriting/output/mmlu_college_medicine_test/", 
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query_rewriting/output/mmlu_medical_genetics_test/", 
            "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/medical_RAG/RAG/query_rewriting/output/mmlu_professional_medicine_test/"
        ]
    }