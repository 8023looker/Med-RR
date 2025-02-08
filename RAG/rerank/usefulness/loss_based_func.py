import json
import ujson
import numpy as np
import copy
import os
from hashlib import md5
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import RobertaModel
from accelerate import Accelerator, skip_first_batches
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import Trainer

def prepare_batch(batch, device=torch.device("cuda:0")): # device=torch.device("cuda:0")
    """ Move the batch to the device. """
    for key in batch: # 1 batch contains 1 item
        batch[key] = batch[key].to(device)

    
class LossTrainer(Trainer):
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)
    
    def loss_based_evaluation(self, 
                               dataloader,
                               model):
        torch.random.manual_seed(0)  # set the random seed for torch
        device = next(model.parameters()).device
        print("Device:", device)

        loss_value = self.obtain_loss(data_loader, model)
     
        model.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("Finished")
        return loss_value
    
       
    def obtain_loss(self, dataloader: torch.utils.data.DataLoader,
                    model: torch.nn.Module,): # put all tensors on the same device
        """ Get the loss of the model on the given dataset. """
        # total_loss = 0
        # total_tokens = 0
        loss_list = []
        for batch in tqdm(dataloader):
            prepare_batch(batch)
            num_token = (batch["labels"] != -100).sum()
            with torch.inference_mode():
                loss = model(**batch).loss * num_token
            # total_loss += loss.item()
            # total_tokens += num_token.item()
            item_loss_value = loss.item() / num_token.item()
            loss_list.append(item_loss_value)

        # print(f"Loss: {total_loss / total_tokens}")
        # result = {"num_tokens": total_tokens, "loss": (
        #     total_loss / total_tokens)}
        # with open(os.path.join(output_dir, "loss.txt"), "w") as f:
        #     f.write(json.dumps(result, indent=4))
        return loss_list
    