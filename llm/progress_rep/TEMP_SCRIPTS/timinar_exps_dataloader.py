#0. INITIAL SET-UP -----------------------------------------------------------------------------------------------

#load libraries
import torch
import wandb
import math
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import load_dataset, Dataset
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import re

#set model params
num_epochs = 1
model_name = "timinar/baby-llama-58m" 
use_subset = True
subset_size = 20
batch_size = 1
gradient_accum_steps = 8
logging_steps = 1
save_total_limit = 1
max_sequence_length = 256

#set file paths
train_file = "llm/data/madlad_from_huggingface/gd_clean_0000.jsonl.gz" 
output_dir = "llm/model_results/llama_finetuned"


#1. LOAD MODEL --------------------------------------------------------------------------------------

#load model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to("cpu")


#2. LOAD AND TOKENIZE DATA --------------------------------------------------------------------------------------

#load json dataset
dataset = load_dataset("json", data_files={"train": train_file}, split="train")


def get_clean_dataset(dataset):
    import re

    def clean_text(example):
        text = example['text'].lower()
        # keep only a-z and whitespace
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # insert full stop every 10 tokens
        tokens = text.split()
        chunks = [tokens[i:i+10] for i in range(0, len(tokens), 20)]
        text_with_periods = ' . '.join([' '.join(chunk) for chunk in chunks])
        
        return {'text': text_with_periods}

    cleaned = dataset.map(clean_text)

    # filter out empty or too short texts
    cleaned = cleaned.filter(lambda x: len(x['text']) > 10)  # example threshold

    return cleaned

dataset = get_clean_dataset(dataset)

#use only a subset of the dataset
if use_subset == True:
    dataset = dataset.select(range(600, 800))

#load tokenizer from model
tokenizer = AutoTokenizer.from_pretrained(model_name)

#add padding token (needed for inputs to be the same size)
tokenizer.add_tokens(["<|pad|>"])
tokenizer.pad_token = "<|pad|>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")

#function for setting tokenizer args
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        # padding=False,
        padding="max_length",
        return_attention_mask = True,
        max_length=max_sequence_length
    )

#apply tokenizer to the dataset
tokenized_subset = dataset.map(tokenize)

#batch sequences and apply padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#remove samples causing index errors
def remove_bad_indices(tokenized_dataset, tokenizer, model):
    bad_indices = []
    model.eval() 
    def is_valid(example, idx):
        try:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to("cpu")
            pad_id = tokenizer.pad_token_id
            attention_mask = (input_ids != pad_id).long()

            labels = input_ids.clone()
            labels[input_ids == pad_id] = -100

            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return True

        except (IndexError, ValueError):
            bad_indices.append(idx)
            return False
        except Exception as e:
            print(f"Other error at index {idx}: {e}")
            bad_indices.append(idx)
            return False
    
    indexed_dataset = tokenized_dataset.add_column("idx", list(range(len(tokenized_dataset))))
    filtered_dataset = indexed_dataset.filter(lambda x: is_valid(x, x["idx"]))
    filtered_dataset = filtered_dataset.remove_columns("idx")

    print(f"Filtered out {len(bad_indices)} examples due to model input errors.")
    print("Bad indices:", bad_indices)

    return filtered_dataset, bad_indices

tokenized_subset, bad_indices = remove_bad_indices(tokenized_subset, tokenizer, model)


print("starting training")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
model.to("cpu")

for epoch in range(num_epochs):
    print(f"\nStarting epoch {epoch+1}/{num_epochs}")
    for i, example in enumerate(tokenized_subset):
        try:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to("cpu")
            pad_id = tokenizer.pad_token_id
            attention_mask = (input_ids != pad_id).long()

            labels = input_ids.clone()
            labels[input_ids == pad_id] = -100

            # Forward pass with labels for loss calculation
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()  # compute gradients
            optimizer.step()
            model.zero_grad()

            print(f"Sample {i} loss: {loss.item()}")

            model.zero_grad()  # reset gradients before next step

        except IndexError:
            print(f"IndexError encountered at sample {i}, skipping.")
            continue
        except Exception as e:
            print(f"Unexpected error at sample {i}: {e}, skipping.")
            continue
