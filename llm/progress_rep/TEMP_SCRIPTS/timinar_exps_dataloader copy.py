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
    # dataset = dataset.select(range(subset_size))
        # dataset = dataset.select(range(1096, 1097))
    # dataset = dataset.select(range(649, 650))
    # dataset = dataset.select(range(645, 646))
    dataset = dataset.select(range(2000))

for i, example in enumerate(dataset):
    if 'text' not in example:
        print(f"Missing 'text' at index {i}")
    elif not isinstance(example['text'], str):
        print(f"Non-string 'text' at index {i}: {example['text']}")

print(f"dataset type: {type(dataset)}")
print(f"dataset len: {len(dataset)}")

for i, example in enumerate(dataset):
    print(i, type(example), len(example), example, "\n")

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

tok_output = tokenizer(dataset[0]['text'], return_attention_mask=True)
print("Tokenized input_ids:", tok_output['input_ids'])
print("Tokenized tokens:", tokenizer.convert_ids_to_tokens(tok_output['input_ids']))
print("Attention mask:", tok_output['attention_mask'])
print("Length tokens:", len(tok_output['input_ids']))


#3. REPORT INITIAL LOSS ------------------------------------------------------------------------------------------

#function for computing cross entropy loss
def compute_loss(model, dataset):
    model.eval()
    losses = []
    with torch.no_grad():
        for example in dataset:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to("cpu")
            print(f"input ids: {input_ids}")
            print(f"input ids len: {input_ids.size(1)}")
            pad_id = int(tokenizer.pad_token_id)
            attention_mask = (input_ids != pad_id).long()

            labels = input_ids.clone()
            labels[input_ids == tokenizer.pad_token_id] = -100

            print(f"input_ids shape: {input_ids.shape}")
            print(f"attention_mask shape: {attention_mask.shape}")
            print(f"labels shape: {labels.shape}")
            print(f"Non-masked tokens: {(labels != -100).sum().item()}")

            if (labels != -100).sum().item() == 0:
                continue

            print(f"labels: {labels}")
            print(f"labels len: {labels.size(1)}")

            assert input_ids.shape == labels.shape, f"Shapes don't match: input_ids {input_ids.shape}, labels {labels.shape}"

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            print("outputs complete")

            loss = outputs.loss.item()

            print(f"Loss: {loss}")

            if not math.isnan(loss):
                losses.append(loss)

    #failsafe if length is zero
    if len(losses) == 0:
        return float('nan'), float('nan')

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

# #compute the loss and perplexity before fine-tuning
# initial_loss, initial_ppl = compute_loss(model, tokenized_subset)

# print("\n--- Before Training ---")
# print(f"Cross-Entropy Loss: {initial_loss:.4f}")
# print(f"Perplexity: {initial_ppl:.2f}")


#4. MODEL TRAINING ----------------------------------------------------------------------------------------------

# #set training args
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     per_device_train_batch_size=batch_size,
#     gradient_accumulation_steps=gradient_accum_steps,
#     num_train_epochs=num_epochs,
#     logging_dir="./logs",
#     logging_steps=logging_steps,
#     save_total_limit=save_total_limit,
#     fp16=False
# )

# #set up model trainger
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_subset,
#     eval_dataset = tokenized_subset,
#     tokenizer=tokenizer,
#     data_collator=data_collator
# )

# #train the model
# trainer.train()

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

#6. FINISHING UP --------------------------------------------------------------------------------------------

# #report final loss
# model.to("cpu")
# final_loss, final_ppl = compute_loss(model, tokenized_subset)

# print("\n--- After Training ---")
# print(f"Cross-Entropy Loss: {final_loss:.4f}")
# print(f"Perplexity: {final_ppl:.2f}")