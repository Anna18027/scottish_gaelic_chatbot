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
from train_functions import remove_bad_indices, is_not_empty, get_elapsed_time, compute_loss

def load_train_val_data(args):
    #load data
    train_dataset = load_dataset("json", data_files={"train": args.train_file}, split="train")
    val_dataset = load_dataset("text", data_files={"validation": args.val_file}, split="validation")
    
    #filter out empty strings
    train_dataset = train_dataset.filter(is_not_empty)
    val_dataset = val_dataset.filter(is_not_empty)

    return train_dataset, val_dataset

def tokenize_data(train_dataset, val_dataset, tokenizer, model, device, args):
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
