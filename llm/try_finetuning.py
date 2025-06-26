#0. INITIAL SET-UP -----------------------------------------------------------------------------------------------

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
from itertools import islice
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

#set model params
num_epochs = 6
model_name = "timinar/baby-llama-58m"
subset_size = 200

#set wandb run name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"test-{timestamp}"

class EpochLossLoggerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        last_epoch = None
        train_loss = None
        eval_loss = None

        # Walk log history backwards to find the most recent losses
        for log in reversed(state.log_history):
            if "epoch" in log:
                if "loss" in log:
                    train_loss = log["loss"]
                    last_epoch = log["epoch"]
                if "eval_loss" in log:
                    eval_loss = log["eval_loss"]
                    last_epoch = log["epoch"]
                if train_loss is not None and eval_loss is not None:
                    break

        if last_epoch is not None:
            wandb.log({
                "epoch": last_epoch,
                "train_loss": train_loss,
                "val_loss": eval_loss
            })

# Initialize wandb
wandb.init(
    project="tinyllama-finetune",
    name=run_name,
    config={
        "model": model_name,
        "epochs": num_epochs,
        "batch_size": 1,
        "grad_accum": 8,
        "max_length": 128
    }
)

# Choose model and training files
# model_name = "timinar/baby-llama-58m" #"TinyLlama/TinyLlama_v1.1"
train_file = "llm/data/madlad_from_huggingface/gd_clean_0000.jsonl.gz" #"llm/data/temp_data/gaidhlig_test_set.txt"
val_file = "llm/data/temp_data/gaidhlig_test_set.txt"
# val_file = "llm/data/temp_data/english_test_set.txt"
output_dir = "llm/model_results/llama_finetuned"

#1. LOAD TOKENIZER & MODEL --------------------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = "<|pad|>"

model = AutoModelForCausalLM.from_pretrained(model_name)
model.to("cpu")

#2. LOAD AND TOKENIZE DATA --------------------------------------------------------------------------------------

# def tokenize(example):
#     return tokenizer(
#         example["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     )

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding=False,
        return_attention_mask = True
    )

def is_not_empty(example):
    return "text" in example and bool(example["text"].strip())
    # return bool(example["text"].strip())

# dataset = load_dataset("text", data_files={"train": train_file}, streaming=True)
dataset = load_dataset("json", data_files={"train": train_file}, split="train")
# dataset = dataset["train"].filter(is_not_empty)
dataset = dataset.filter(is_not_empty)
subset = dataset.select(range(subset_size))
tokenized_subset = subset.map(tokenize)

# Load and tokenize validation set (assumes val_file is plain text or JSON lines)
val_dataset_raw = load_dataset("text", data_files={"validation": val_file}, split="validation")
val_dataset = val_dataset_raw.filter(is_not_empty)
tokenized_val = val_dataset.map(tokenize)

# #subsetting - to remove later
# tokenized_subset = list(islice(tokenized_dataset, 2))
# tokenized_subset = Dataset.from_list(tokenized_subset)

print(f"Number of training examples: {len(tokenized_subset)}")
for i, example in enumerate(tokenized_subset):
    print(f"\n[Example {i}] Input token IDs:\n{example['input_ids']}")
    print(f"[Example {i}] Decoded text:\n{tokenizer.decode(example['input_ids'], skip_special_tokens=False)}")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Function to compute average cross-entropy loss and perplexity
# def compute_loss(model, dataset):
#     model.eval()
#     losses = []
#     with torch.no_grad():
#         for example in dataset:
#             input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to("cpu")
#             attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to("cpu")
#             labels = input_ids.clone()
#             labels[input_ids == tokenizer.pad_token_id] = -100
            
#             # Skip examples where all tokens are masked (no loss to compute)
#             if (labels != -100).sum().item() == 0:
#                 continue
            
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss.item()
#             if not math.isnan(loss):
#                 losses.append(loss)

#     if len(losses) == 0:
#         return float('nan'), float('nan')

#     avg_loss = sum(losses) / len(losses)
#     perplexity = math.exp(avg_loss)
#     return avg_loss, perplexity
def compute_loss(model, dataset):
    model.eval()
    losses = []
    with torch.no_grad():
        for example in dataset:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to("cpu")

            # Create attention mask manually if not in dataset
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            labels = input_ids.clone()
            labels[input_ids == tokenizer.pad_token_id] = -100

            # Skip if all tokens are ignored
            if (labels != -100).sum().item() == 0:
                continue

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.item()
            if not math.isnan(loss):
                losses.append(loss)

    if len(losses) == 0:
        return float('nan'), float('nan')

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


#3. REPORT INITIAL LOSS ------------------------------------------------------------------------------------------

initial_loss, initial_ppl = compute_loss(model, tokenized_subset)
print("\n--- Before Training ---")
print(f"Cross-Entropy Loss: {initial_loss:.4f}")
print(f"Perplexity: {initial_ppl:.2f}")

# ✅ Log to wandb
wandb.log({
    "initial_loss": initial_loss,
    "initial_perplexity": initial_ppl
})

#4. TRAINING SETUP ----------------------------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=num_epochs,
    logging_dir="./logs",
    logging_steps=1,
    save_total_limit=1,
    fp16=False,
    report_to="wandb",
)

#5. TRAIN --------------------------------------------------------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_subset,
    eval_dataset = tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks = [EpochLossLoggerCallback()]
)

trainer.train()

#log epoch losses
epoch_train_losses = []
epoch_val_losses = []

for log in trainer.state.log_history:
    if "epoch" in log:
        epoch = log["epoch"]
        if "loss" in log:
            epoch_train_losses.append((epoch, log["loss"]))
        if "eval_loss" in log:
            epoch_val_losses.append((epoch, log["eval_loss"]))

# Save losses
loss_file = os.path.join(output_dir, "epoch_losses.json")
with open(loss_file, "w") as f:
    json.dump({
        "train_loss": epoch_train_losses,
        "val_loss": epoch_val_losses
    }, f)

# Plot the loss curve
train_epochs = [x[0] for x in epoch_train_losses]
train_losses = [x[1] for x in epoch_train_losses]

val_epochs = [x[0] for x in epoch_val_losses]
val_losses = [x[1] for x in epoch_val_losses]

plt.figure(figsize=(6, 4))
plt.plot(train_epochs, train_losses, marker='o', label="Train Loss")
plt.plot(val_epochs, val_losses, marker='s', label="Validation Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
plt.close()

print(f"Saved loss curve to {output_dir}/loss_curve.png")
print(f"Saved raw loss values to {loss_file}")

#6. REPORT FINAL LOSS --------------------------------------------------------------------------------------------

model.to("cpu")
final_loss, final_ppl = compute_loss(model, tokenized_subset)
print("\n--- After Training ---")
print(f"Cross-Entropy Loss: {final_loss:.4f}")
print(f"Perplexity: {final_ppl:.2f}")

# ✅ Log to wandb
wandb.log({
    "final_loss": final_loss,
    "final_perplexity": final_ppl
})

#7. SAVE MODEL ---------------------------------------------------------------------------------------------------

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

#8. FINISH WANDB RUN --------------------------------------------------------------------------------------------

wandb.finish()
print("Training completed.")
