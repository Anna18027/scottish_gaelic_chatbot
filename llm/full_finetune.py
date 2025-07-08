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
from datasets import load_dataset
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from peft import get_peft_model, LoraConfig, TaskType

#set model params
num_epochs = 1
model_name = "meta-llama/Llama-3.2-1B"  #"timinar/baby-llama-58m" #"meta-llama/Llama-3.2-1B"
use_subset = True
subset_size = 2000
batch_size = 1
gradient_accum_steps = 1 #8
logging_steps = 1
save_total_limit = 1

#set file paths
train_file = "/exports/eddie/scratch/s2751141/data/madlad_from_huggingface/gd_clean_0000.jsonl.gz"
val_file = "/exports/eddie/scratch/s2751141/data/temp_data/gaidhlig_test_set.txt"

#train_file = "llm/data/madlad_from_huggingface/gd_clean_0000.jsonl.gz" #"llm/data/temp_data/gaidhlig_test_set.txt"
#val_file = "llm/data/temp_data/gaidhlig_test_set.txt" #"llm/data/temp_data/english_test_set.txt"
output_dir = "llm/model_results/llama_finetuned"

#set wandb run name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"test-{timestamp}"

#set up wandb params
wandb.init(
    project="tinyllama-finetune",
    name=run_name,
    config={
        "model": model_name,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "grad_accum": gradient_accum_steps
    }
)

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#1. LOAD TOKENIZER & MODEL --------------------------------------------------------------------------------------


#os.environ["HF_HOME"] = "/exports/eddie/scratch/s2751141/hf_cache"
#os.environ["TRANSFORMERS_CACHE"] = "/exports/eddie/scratch/s2751141/hf_cache/transformers"
#os.environ["HF_DATASETS_CACHE"] = "/exports/eddie/scratch/s2751141/hf_cache/datasets"
print("below is hf token from python:")
print(os.environ.get("HUGGINGFACE_HUB_TOKEN"))
print("end of token")

model_path = "/exports/eddie/scratch/s2751141/hf_models/models--meta-llama--Llama-3.2-1B"

print("starting to load model")


print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
print("Cached:", torch.cuda.memory_reserved() / 1e9, "GB")
torch.cuda.empty_cache()
print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
print("Cached:", torch.cuda.memory_reserved() / 1e9, "GB")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)


print("tokenizer loaded")
print("Vocab size before adding pad token:", len(tokenizer))

print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
print("Cached:", torch.cuda.memory_reserved() / 1e9, "GB")


# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)
# .to("cuda" if torch.cuda.is_available() else "cpu")

print("model loaded")

print("Before gradient checkpointing enabled:", model.is_gradient_checkpointing)
model.gradient_checkpointing_disable()  # just in case, to reset
model.gradient_checkpointing_enable()
print("enabled gradient checkpointing")
print("Gradient checkpointing enabled:", model.is_gradient_checkpointing)

#peft_config = LoraConfig(
#    r=4,
#    lora_alpha=64,
#    target_modules=[
#    "q_proj", "v_proj",
#    "k_proj", "o_proj",             
#    "gate_proj", "up_proj", "down_proj"
#],
#    lora_dropout=0.05,
#    bias="none",
#    task_type=TaskType.CAUSAL_LM
#)
#model = get_peft_model(model, peft_config)


print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
print("Cached:", torch.cuda.memory_reserved() / 1e9, "GB")


#load tokenizer from model
#tokenizer = AutoTokenizer.from_pretrained(model_name)

#add padding token (needed for inputs to be the same size)
tokenizer.add_tokens(["<|pad|>"])
tokenizer.pad_token = "<|pad|>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
print("Vocab size after adding pad token:", len(tokenizer))

print("padding tokens added")

old_embed_size = model.get_input_embeddings().weight.shape[0]
print("Model embedding size before resize:", old_embed_size)
model.resize_token_embeddings(len(tokenizer))
new_embed_size = model.get_input_embeddings().weight.shape[0]
print("Model embedding size after resize:", new_embed_size)

#function for setting tokenizer args
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding=False,
        return_attention_mask = True
    )

#function for stripping empty sentences
def is_not_empty(example):
    return "text" in example and bool(example["text"].strip())
    # return bool(example["text"].strip())

#load model
#model = AutoModelForCausalLM.from_pretrained(model_name)
#model.to("cpu")


#2. LOAD AND TOKENIZE DATA --------------------------------------------------------------------------------------

#load simple text dataset (comment if using json file instead)
# dataset = load_dataset("text", data_files={"train": train_file}, streaming=True)

print("starting to load data")

print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
print("Cached:", torch.cuda.memory_reserved() / 1e9, "GB")

#load json dataset
dataset = load_dataset("json", data_files={"train": train_file}, split="train")

print("data loaded")

#remove empty lines
dataset = dataset.filter(is_not_empty)

#use only a subset of the dataset
if use_subset == True:
    dataset = dataset.select(range(subset_size))

#apply tokenizer to the dataset
tokenized_subset = dataset.map(tokenize)


lengths = [len(x['input_ids']) for x in tokenized_subset]
print(f"Max sequence length: {max(lengths)}")
print(f"Average sequence length: {sum(lengths) / len(lengths):.2f}")

#load and tokenise the validation set
val_dataset_raw = load_dataset("text", data_files={"validation": val_file}, split="validation")
val_dataset = val_dataset_raw.filter(is_not_empty)
tokenized_val = val_dataset.map(tokenize)

#print first few examples for debugging
#print(f"Number of training examples: {len(tokenized_subset)}")
#for i, example in enumerate(tokenized_subset):
#    print(f"\n[Example {i}] Input token IDs:\n{example['input_ids']}")
#    print(f"[Example {i}] Decoded text:\n{tokenizer.decode(example['input_ids'], skip_special_tokens=False)}")

#batch sequences and apply padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


#3. REPORT INITIAL LOSS ------------------------------------------------------------------------------------------

#set class for loss logging
class EpochLossLoggerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        #initialise
        last_epoch = None
        train_loss = None
        eval_loss = None

        #get most recent losses
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

        #log with wandb
        if last_epoch is not None:
            wandb.log({
                "epoch": last_epoch,
                "train_loss": train_loss,
                "val_loss": eval_loss
            })

#function for computing cross entropy loss
def compute_loss(model, dataset):
    model.eval()
    losses = []
    with torch.no_grad():
        for example in dataset:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            pad_id = int(tokenizer.pad_token_id)
            attention_mask = (input_ids != pad_id).long()

            labels = input_ids.clone()
            labels[input_ids == tokenizer.pad_token_id] = -100

            if (labels != -100).sum().item() == 0:
                continue

            outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
            loss = outputs.loss.item()
            if not math.isnan(loss):
                losses.append(loss)

    #failsafe if length is zero
    if len(losses) == 0:
        return float('nan'), float('nan')

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

#compute the loss and perplexity before fine-tuning
initial_loss, initial_ppl = compute_loss(model, tokenized_subset)

print("\n--- Before Training ---")
print(f"Cross-Entropy Loss: {initial_loss:.4f}")
print(f"Perplexity: {initial_ppl:.2f}")

wandb.log({
    "initial_loss": initial_loss,
    "initial_perplexity": initial_ppl
})


#4. MODEL TRAINING ----------------------------------------------------------------------------------------------

#set training args
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accum_steps,
    num_train_epochs=num_epochs,
    logging_dir="./logs",
    logging_steps=logging_steps,
    save_total_limit=save_total_limit,
    #fp16=True if torch.cuda.is_available() else False,
    fp16=torch.cuda.is_available(),
    bf16=False,
    report_to="wandb",
)

#set up model trainger
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_subset,
    eval_dataset = tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks = [EpochLossLoggerCallback()]
)

print("Using device:", next(model.parameters()).device)

#train the model
trainer.train()


#5. SAVE LOSSES PER EPOCH ----------------------------------------------------------------------------------------------

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

#save losses
loss_file = os.path.join(output_dir, "epoch_losses.json")
with open(loss_file, "w") as f:
    json.dump({
        "train_loss": epoch_train_losses,
        "val_loss": epoch_val_losses
    }, f)

#plot loss curve
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

#save plot to file
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
plt.close()

print(f"Saved loss curve to {output_dir}/loss_curve.png")
print(f"Saved raw loss values to {loss_file}")


#6. FINISHING UP --------------------------------------------------------------------------------------------

#report final loss
model.to(device)
final_loss, final_ppl = compute_loss(model, tokenized_subset)
print("\n--- After Training ---")
print(f"Cross-Entropy Loss: {final_loss:.4f}")
print(f"Perplexity: {final_ppl:.2f}")

#log final loss to wandb
wandb.log({
    "final_loss": final_loss,
    "final_perplexity": final_ppl
})

#save model 
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

#finish wandb run
wandb.finish()
print("Training completed.")
