#0. INITIAL SET-UP -----------------------------------------------------------------------------------------------
from datetime import datetime; start_time = datetime.now()

#load libraries
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
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from finetune_args import parse_args
from finetune_functions import remove_bad_indices, is_not_empty, get_elapsed_time, compute_loss

#import model args
args = parse_args()

#create results directory for each run
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_output_dir = os.path.join(args.output_dir, f"results_{timestamp}")
os.makedirs(results_output_dir, exist_ok=True)

#use gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print progress 
print(f"0. Initial set-up is complete. Total time taken: {get_elapsed_time(start_time)}")


#1. LOAD TOKENIZER & MODEL --------------------------------------------------------------------------------------

#load tokenizer from model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

#add padding token (needed for inputs to be the same size)
tokenizer.add_tokens(["<|pad|>"])
tokenizer.pad_token = "<|pad|>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")

#load model
model = AutoModelForCausalLM.from_pretrained(args.model_name)
model.to(device)
model.resize_token_embeddings(len(tokenizer), mean_resizing=True)

#print progress 
print(f"1. Load tokenizer and model is complete. Total time taken:{get_elapsed_time(start_time)}")


#2. LOAD AND TOKENIZE DATA --------------------------------------------------------------------------------------

#load json dataset
dataset = load_dataset("json", data_files={"train": args.train_file}, split="train")
print("data loaded")
print(f"length of dataset before filtering for empty:{len(dataset)}")

#remove empty lines
dataset = dataset.filter(is_not_empty)
print(f"length of dataset after filtering for empty:{len(dataset)}")

#get total tokens before subsetting
texts = list(dataset["text"])
batch_encoding = tokenizer(texts, truncation=False, padding=False)
total_tokens = sum(len(ids) for ids in batch_encoding["input_ids"])

#use only a subset of the dataset
if args.subset_size>0:
    dataset = dataset.select(range(args.subset_size))

#get total tokens after subsetting
texts = list(dataset["text"])
batch_encoding = tokenizer(texts, truncation=False, padding=False)
subset_tokens = sum(len(ids) for ids in batch_encoding["input_ids"])

#compute proportion of tokens in subset compared to full dataset
prop_tokens = subset_tokens/total_tokens
print(f"number of tokens in dataset after subsetting: {subset_tokens}")
print(f"proportion of tokens in dataset after subsetting: {prop_tokens}")

#function for setting tokenizer args
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        return_attention_mask = True,
        max_length=args.max_sequence_length
    )

#apply tokenizer to the dataset
tokenized_subset = dataset.map(tokenize, batched=True, batch_size=1000)

#load and tokenise the validation set
val_dataset_raw = load_dataset("text", data_files={"validation": args.val_file}, split="validation")
val_dataset = val_dataset_raw.filter(is_not_empty)
tokenized_val = val_dataset.map(tokenize, batched=True, batch_size=1000)

#remove data samples causing errors (should be very small proportion)
tokenized_subset, bad_indices_train = remove_bad_indices(tokenized_subset, tokenizer, model, device)
tokenized_val, bad_indices_val = remove_bad_indices(tokenized_val, tokenizer, model, device)

#batch sequences and apply padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#print progress 
print(f"2. Load and tokenize data is complete. Total time taken:{get_elapsed_time(start_time)}")


#3. REPORT INITIAL LOSS ------------------------------------------------------------------------------------------

#compute the loss and perplexity before fine-tuning
initial_loss, initial_ppl = compute_loss(model, tokenized_val, tokenizer, device)

print("\n--- Before Training ---")
print(f"Cross-Entropy Loss: {initial_loss:.4f}")
print(f"Perplexity: {initial_ppl:.2f}")

#print progress 
print(f"3. Report initial loss is complete. Total time taken:{get_elapsed_time(start_time)}")


#4. MODEL TRAINING ----------------------------------------------------------------------------------------------

#set training args
training_args = TrainingArguments(
    output_dir=args.output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accum_steps,
    num_train_epochs=args.num_epochs,
    logging_dir="./logs",
    save_total_limit=args.save_total_limit,
    fp16=False,
    report_to=None,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False #lower loss is better,
)

#set up model trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_subset,
    eval_dataset = tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator
)

#train the model
trainer.train()

#print progress 
print(f"4. Model training is complete. Total time taken:{get_elapsed_time(start_time)}")


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
loss_file = os.path.join(results_output_dir, "epoch_losses.json")
with open(loss_file, "w") as f:
    json.dump({
        "train_loss": epoch_train_losses,
        "val_loss": epoch_val_losses
    }, f)

best_epoch_idx = min(range(len(epoch_val_losses)), key=lambda i: epoch_val_losses[i][1])
best_epoch_num, best_val_loss = epoch_val_losses[best_epoch_idx]

print(f"Best Epoch: {best_epoch_num:.1f} with Validation Loss: {best_val_loss:.4f}")

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
plt.savefig(os.path.join(results_output_dir, "loss_curve.png"))
plt.close()

print(f"Saved loss curve to {results_output_dir}/loss_curve.png")
print(f"Saved raw loss values to {loss_file}")

#print progress 
print(f"5. Save losses per epoch is complete. Total time taken:{get_elapsed_time(start_time)}")


#6. FINISHING UP --------------------------------------------------------------------------------------------

#report final loss
model.to(device)
final_loss, final_ppl = compute_loss(model, tokenized_val, tokenizer, device)
print("\n--- After Training ---")
print(f"Cross-Entropy Loss: {final_loss:.4f}")
print(f"Perplexity: {final_ppl:.2f}")

#save model 
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print("Training completed.")


#ADD GENERATION FROM HALF PROMPTS ---------------------------------------------------------------------------------------------

#set file path to finetuned model (output from top half of script)
finetuned_path = args.output_dir

#load fine-tuned model
finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_path)
finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
finetuned_model.to(device)
finetuned_model.eval()

#set up Gaelic and English prompts
prompts = [
    "Tha mi a’ smaoineachadh gu bheil e",
    "Chuala mi an naidheachd",
    "Bha an latha ro",
    "Thàinig iad gu",
    "Tha mi a’ dol a",
]

english_prompts = [
    "I think that it is",
    "I heard the news",
    "The day was too",
    "They came to",
    "I am going to",
]

print("\n--- Sentence Completions ---\n")

gaelic_completions = []
english_completions = []

for i, (gaelic_prompt, english_prompt) in enumerate(zip(prompts, english_prompts)):
    print(f"[{i}] GAELIC PROMPT:  {gaelic_prompt}")
    print(f"    ENGLISH PROMPT: {english_prompt}")

    # Gaelic prompt completion
    input_ids_gaelic = finetuned_tokenizer(gaelic_prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output_gaelic = finetuned_model.generate(
            input_ids=input_ids_gaelic,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            pad_token_id=finetuned_tokenizer.pad_token_id,
            eos_token_id=finetuned_tokenizer.eos_token_id,
        )
    decoded_gaelic = finetuned_tokenizer.decode(output_gaelic[0], skip_special_tokens=True)
    completion_gaelic = decoded_gaelic[len(gaelic_prompt):].strip()
    print(f"    [GAELIC COMPLETION]:  {completion_gaelic}")

    # English prompt completion
    input_ids_english = finetuned_tokenizer(english_prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output_english = finetuned_model.generate(
            input_ids=input_ids_english,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            pad_token_id=finetuned_tokenizer.pad_token_id,
            eos_token_id=finetuned_tokenizer.eos_token_id,
        )
    decoded_english = finetuned_tokenizer.decode(output_english[0], skip_special_tokens=True)
    completion_english = decoded_english[len(english_prompt):].strip()
    print(f"    [ENGLISH COMPLETION]: {completion_english}\n")

    # Save each
    gaelic_completions.append({
        "prompt": gaelic_prompt,
        "completion": completion_gaelic
    })

    english_completions.append({
        "prompt": english_prompt,
        "completion": completion_english
    })

# Save metadata
metadata = {
    "timestamp": timestamp,
    "model": args.model_name,
    "batch_size": args.batch_size,
    "gradient_accumulation_steps": args.gradient_accum_steps,
    "num_epochs": args.num_epochs,
    "max_sequence_length": args.max_sequence_length,
    "subset_size": args.subset_size if args.subset_size>0 else "full dataset",
    "train_tokens": subset_tokens,
    "token_proportion": prop_tokens,
    "best_epoch": best_epoch_num,
    "best_val_loss": best_val_loss,
    "best_loss": final_loss,
    "best_perplexity": final_ppl,
    "bad_indices_train": bad_indices_train,
    "bad_indices_val": bad_indices_val,
    "max_new_tokens": args.max_new_tokens,
    "temperature": args.temperature,
    "top_p": args.top_p,
    "do_sample": args.do_sample,
    "gaelic_completions": gaelic_completions,
    "english_completions": english_completions
}

metadata_file = os.path.join(results_output_dir, "metadata.json")
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print(f"Saved training metadata to {metadata_file}")

#print progress 
print(f"6. Finishing up and generation is complete. Total time taken: {get_elapsed_time(start_time)}")

