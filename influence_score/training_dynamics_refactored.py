import argparse
from collator import DataCollatorForPairWithPadding
from data_utils import (
    load_selection_dataset,
    preprocess_function,
)
from scores import (
    get_sequence_logps
)

from datasets import load_dataset, Dataset
from functools import partial
import json
import numpy as np
import pandas as pd
from peft import PeftConfig, PeftModel
import os
import torch
import torch.multiprocessing as mp
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    DataCollatorWithPadding,
    default_data_collator
)
from tqdm import tqdm
from typing import Union


def load_dataset_and_tokenizer(
    dataset_path: str,
    model_name_or_path: str
):
    """
    Load dataset and tokenizer.
    """
    
    # load dataset
    ds = load_selection_dataset(
        dataset_path=dataset_path,
        cached=True
    )
    
    # ds = ds.filter(lambda x: x['benign'] == False and x['lang'] == 'python')
    # ds = ds.filter(lambda x: x['benign'] == True and x['lang'] == 'python')
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    assert tokenizer.padding_side == 'left'
    
    return ds, tokenizer


def load_model_and_optimizer(
    base_model_name_or_path,
    checkpoint_dir,
    merge_and_unload=False
):
    """
    load model and optimizer
    """

    # load hf model
    peft_config = PeftConfig.from_pretrained(
        checkpoint_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        # trust_remote_code=True,   # pickle issues with mp
        torch_dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="flash_attention_2"
    )
    peft_model = PeftModel.from_pretrained(
        model=model,
        model_id=checkpoint_dir,
        config=peft_config,
    )
    if merge_and_unload:
        peft_model = peft_model.merge_and_unload()
        peft_model.eval()

    return peft_model, None

    
def checkpoint_stats(
    rank: int,
    world_size: int,
    checkpoint_id: Union[str, int],
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    batch_size: int = 4,
    optimizer: torch.optim.Optimizer = None,
    output_dir: str = None,
):
    """
    get the stats of a checkpoint (support multi-gpu)
    
    Args:
    - checkpoint_dir: str, path to the checkpoint
    - dataset: huggingface dataset, the dataset used for data selection
    """
    
    # setup
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    column_names = dataset.column_names
    processed_ds = dataset.shard(num_shards=world_size, index=rank).map(
        partial(preprocess_function, tokenizer=tokenizer), 
        batched=False,  # batching will cause NaN in the logps (too many paddings)
        remove_columns=column_names,
        num_proc=16
    )
    collator = DataCollatorForPairWithPadding(
        tokenizer=tokenizer, padding='longest', return_tensors='pt')
    dataloader = torch.utils.data.DataLoader(
        processed_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        # num_workers=4,
        num_workers=2,
        pin_memory=True
    )
    
    stats = {}
    # for i, batch in enumerate(dataloader):
    for i, batch in enumerate(tqdm(dataloader, desc=f'Processing shard {rank}')):
        
        # if i > 500: # debug
        # if i > 200: # debug
        #     break
        
        MAX_LENGTH = 2048   # NOTE: might by problematic
        chosen_input_ids = batch['chosen_input_ids'][:,:MAX_LENGTH].to(device)
        chosen_labels = batch['chosen_labels'][:,:MAX_LENGTH].to(device)
        rejected_input_ids = batch['rejected_input_ids'][:,:MAX_LENGTH].to(device)
        rejected_labels = batch['rejected_labels'][:,:MAX_LENGTH].to(device)
        
        with torch.no_grad():
        
            chosen_logps, chosen_ln_logps, \
                chosen_seq_len, chosen_acc = get_sequence_logps(
                model=model,
                input_ids=chosen_input_ids,
                labels=chosen_labels
            )
            rejected_logps, rejected_ln_logps, \
                rejected_seq_len, rejected_acc = get_sequence_logps(
                model=model,
                input_ids=rejected_input_ids,
                labels=rejected_labels
            )
            
            if 'chosen_logps' not in stats:
                stats['chosen_logps'] = []
                stats['rejected_logps'] = []
                stats['chosen_ln_logps'] = []
                stats['rejected_ln_logps'] = []
                stats['chosen_seq_len'] = []
                stats['rejected_seq_len'] = []
                stats['chosen_acc'] = []
                stats['rejected_acc'] = []
                stats['benign'] = []
            
            stats['chosen_logps'].extend(chosen_logps.cpu().to(torch.float32).numpy().tolist())
            stats['rejected_logps'].extend(rejected_logps.cpu().to(torch.float32).numpy().tolist())
            stats['chosen_ln_logps'].extend(chosen_ln_logps.cpu().to(torch.float32).numpy().tolist())
            stats['rejected_ln_logps'].extend(rejected_ln_logps.cpu().to(torch.float32).numpy().tolist())
            stats['chosen_seq_len'].extend(chosen_seq_len.cpu().to(torch.float32).numpy().tolist())
            stats['rejected_seq_len'].extend(rejected_seq_len.cpu().to(torch.float32).numpy().tolist())
            stats['chosen_acc'].extend(chosen_acc.cpu().to(torch.float32).numpy().tolist())
            stats['rejected_acc'].extend(rejected_acc.cpu().to(torch.float32).numpy().tolist())
            stats['benign'].extend(batch['benign']) # NOTE: this is a list of bools
    
    # save stats
    with open(os.path.join(output_dir, f'checkpoint-{checkpoint_id}_{rank}.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    

def training_dynamics_stats(
    world_size,
    base_model_name_or_path,
    root_checkpoint_dir,
    tokenizer,
    dataset,
    per_device_batch_size=2,
    output_dir=None
):
    """
    Collect statistics of training dynamics.
    
    Args:
    - root_checkpoint_dir: str, path to the root checkpoint directory
    - dataset: huggingface dataset, the dataset used for data selection
    """

    for checkpoint_dir in os.listdir(root_checkpoint_dir):
        
        checkpoint_dir = os.path.join(root_checkpoint_dir, checkpoint_dir)
        if not os.path.isdir(checkpoint_dir):
            print(f"Skipping {checkpoint_dir}")
            continue
        elif 'checkpoint' not in checkpoint_dir:
            continue
        else:
            checkpoint_id = int(checkpoint_dir.split('-')[-1])
            if checkpoint_id % 100 != 0:
                continue
            # if checkpoint_id in [100, 200, 300, 800, 900]:
            #     continue
            print(f"Processing {checkpoint_dir}")
            
        model, _ = load_model_and_optimizer(
            base_model_name_or_path=base_model_name_or_path,
            checkpoint_dir=checkpoint_dir
        )
        
        # parallel inference
        mp.spawn(
            checkpoint_stats,
            args=(
                world_size,
                checkpoint_id,
                model,
                tokenizer,
                dataset,
                per_device_batch_size,
                None,
                output_dir
            ),
            nprocs=world_size,
            join=True
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--base_model_name_or_path', type=str, required=True)
    parser.add_argument('--root_ckpt_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--world_size', type=int, default=8)
    parser.add_argument('--per_device_batch_size', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load dataset and tokenizer
    dataset, tokenizer = load_dataset_and_tokenizer(
        args.dataset_path,
        args.base_model_name_or_path
    )
    if 'clm' in args.base_model_name_or_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.padding_side == 'left'

    # training dynamics
    training_dynamics_stats(
        world_size=args.world_size,
        base_model_name_or_path=args.base_model_name_or_path,
        root_checkpoint_dir=args.root_ckpt_dir,
        tokenizer=tokenizer,
        dataset=dataset,
        per_device_batch_size=args.per_device_batch_size,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
