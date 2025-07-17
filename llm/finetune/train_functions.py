import torch
import math
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import load_dataset
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence

#function to identify and remove data samples which are causing errors
# def remove_bad_indices(tokenized_dataset, tokenizer, model, device):
#     bad_indices = []
#     model.eval() 
#     def is_valid(example, idx):
#         try:
#             input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
#             pad_id = tokenizer.pad_token_id
#             attention_mask = (input_ids != pad_id).long()

#             labels = input_ids.clone()
#             labels[input_ids == pad_id] = -100
#             with torch.no_grad():
#                 _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             return True

#         except (IndexError, ValueError):
#             bad_indices.append(idx)
#             return False
#         except Exception as e:
#             print(f"Other error at index {idx}: {e}")
#             bad_indices.append(idx)
#             return False
    
#     indexed_dataset = tokenized_dataset.add_column("idx", list(range(len(tokenized_dataset))))
#     filtered_dataset = indexed_dataset.filter(lambda x: is_valid(x, x["idx"]))
#     filtered_dataset = filtered_dataset.remove_columns("idx")

#     print(f"Filtered out {len(bad_indices)} examples due to model input errors.")
#     print("Bad indices:", bad_indices)

#     return filtered_dataset, bad_indices
def remove_bad_indices(tokenized_dataset, tokenizer, model, device, batch_size=8):
    model.eval()
    bad_indices = set()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Add indices so we can track which rows cause problems
    tokenized_dataset = tokenized_dataset.add_column("idx", list(range(len(tokenized_dataset))))
    
    # Use DataLoader to batch examples
    loader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

    for batch in tqdm(loader, desc="Checking batches"):
        input_ids_list = []
        attention_mask_list = []
        label_list = []
        indices = []

        # Prepare each example
        for example in batch:
            try:
                input_ids = torch.tensor(example["input_ids"])
                attention_mask = (input_ids != pad_id).long()
                labels = input_ids.clone()
                labels[input_ids == pad_id] = -100

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                label_list.append(labels)
                indices.append(example["idx"])

            except Exception as e:
                print(f"Preprocessing error on index {example['idx']}: {e}")
                bad_indices.add(example["idx"])

        try:
            # Pad to same length
            input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id).to(device)
            attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)
            labels_padded = torch.nn.utils.rnn.pad_sequence(label_list, batch_first=True, padding_value=-100).to(device)

            with torch.no_grad():
                _ = model(input_ids=input_ids_padded, attention_mask=attention_mask_padded, labels=labels_padded)

        except Exception as e:
            print(f"Model error on batch with indices {indices}: {e}")
            bad_indices.update(indices)

    # Remove bad examples
    filtered_dataset = tokenized_dataset.filter(lambda x: x["idx"] not in bad_indices)
    filtered_dataset = filtered_dataset.remove_columns("idx")

    print(f"Filtered out {len(bad_indices)} bad examples.")
    return filtered_dataset, list(bad_indices)

#function for stripping empty sentences
def is_not_empty(example):
    return "text" in example and bool(example["text"].strip())

#function to monitor code timings
def get_elapsed_time(start_time, end_time=None):
    if end_time is None:
        end_time = datetime.now()
    elapsed = end_time - start_time
    minutes, seconds = divmod(elapsed.total_seconds(), 60)
    return f"{int(minutes)} minutes {int(seconds)} seconds"

# def collate_to_device(batch):
#     input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
#     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

#     attention_mask = (input_ids != tokenizer.pad_token_id).long()
#     labels = input_ids.clone()
#     labels[input_ids == tokenizer.pad_token_id] = -100

#     return {
#         "input_ids": input_ids.to(device),
#         "attention_mask": attention_mask.to(device),
#         "labels": labels.to(device),
#     }

# def compute_loss(model, dataset, batch_size=16):
#     model.eval()
#     losses = []

#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_to_device)

#     with torch.no_grad():
#         for batch in dataloader:
#             outputs = model(**batch)
#             loss = outputs.loss.item()
#             if not math.isnan(loss):
#                 losses.append(loss)

#     if len(losses) == 0:
#         return float('nan'), float('nan')

#     avg_loss = sum(losses) / len(losses)
#     perplexity = math.exp(avg_loss)
#     return avg_loss, perplexity
def get_collate_fn(tokenizer, device):
    def collate_to_device(batch):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }
    return collate_to_device


def compute_loss(model, dataset, tokenizer, device, batch_size=16):
    model.eval()
    losses = []

    collate_fn = get_collate_fn(tokenizer, device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss.item()
            if not math.isnan(loss):
                losses.append(loss)

    if len(losses) == 0:
        return float('nan'), float('nan')

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity