from datetime import datetime
import os
import json
import math
import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from types import SimpleNamespace
from datasets import load_dataset
from train_functions import remove_bad_indices, is_not_empty, get_elapsed_time, compute_loss, get_json_tokens

def load_data(args):
    #load data
    train_dataset = load_dataset("json", data_files={"train": args.train_file}, split="train")
    val_dataset = load_dataset("text", data_files={"validation": args.val_file}, split="validation")
    
    #filter out empty strings
    train_dataset = train_dataset.filter(is_not_empty)
    val_dataset = val_dataset.filter(is_not_empty)

    return train_dataset, val_dataset

def tokenize_data(train_dataset, val_dataset, tokenizer, args):
    #set up tokenizer args
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            max_length=args.max_sequence_length,
        )

    #tokenize train and val data
    tokenized_train = train_dataset.map(tokenize, batched=True, batch_size=1000)
    tokenized_val = val_dataset.map(tokenize, batched=True, batch_size=1000)

    return tokenized_train, tokenized_val

def process_data(tokenized_train, tokenized_val, tokenizer, model, device, args):
    tokenized_train = tokenized_train.filter(is_not_empty)
    tokenized_val = tokenized_val.filter(is_not_empty)
    
    #subset training data
    if args.subset_size > 0:
        tokenized_train_subset = tokenized_train.select(range(args.subset_size))
    else:
        tokenized_train_subset = tokenized_train

    #get proportion of tokens in used training data
    total_tokens = get_json_tokens(tokenized_train, tokenizer)
    subset_tokens = get_json_tokens(tokenized_train_subset, tokenizer)
    prop_tokens = subset_tokens/total_tokens
    total_subsets = len(tokenized_train) // args.subset_size if args.subset_size > 0 else 1

    print(f"Proportion of MADLAD tokens used is:{prop_tokens}")
    print(f"Total possible subsets of size {args.subset_size}: {total_subsets}")


    #filter out data samples causing errors
    tokenized_train_subset, bad_indices_train = remove_bad_indices(tokenized_train_subset, tokenizer, model, device)
    tokenized_val, bad_indices_val = remove_bad_indices(tokenized_val, tokenizer, model, device)

    #set up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return tokenized_train_subset, tokenized_val, prop_tokens, data_collator, bad_indices_train, bad_indices_val

