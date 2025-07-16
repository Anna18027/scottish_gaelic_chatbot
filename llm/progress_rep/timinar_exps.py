#0. INITIAL SET-UP -----------------------------------------------------------------------------------------------

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
from datetime import datetime

#set model params
num_epochs = 2
model_name = "timinar/baby-llama-58m"
use_subset = True
subset_size = 20
batch_size = 16
gradient_accum_steps = 8
logging_steps = 1
save_total_limit = 1
max_sequence_length = 256

#choose peft
peft={"none", "head only", "LORA only", "LORA and head"}

#set timestamp for file saving
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#set file paths
train_file = "llm/data/madlad_from_huggingface/gd_clean_0000.jsonl.gz" #"llm/data/temp_data/gaidhlig_test_set.txt"
val_file = "llm/data/temp_data/gaidhlig_test_set.txt" #"llm/data/temp_data/english_test_set.txt"
output_dir = "llm/model_results/llama_finetuned"
results_output_dir = os.path.join(output_dir, f"results_{timestamp}")
os.makedirs(results_output_dir, exist_ok=True)

#use gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#1. LOAD TOKENIZER & MODEL --------------------------------------------------------------------------------------

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

#function for stripping empty sentences
def is_not_empty(example):
    return "text" in example and bool(example["text"].strip())
    # return bool(example["text"].strip())

#load model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.resize_token_embeddings(len(tokenizer))


#check max token length for model
print("model max tokens:")
print(model.config.max_position_embeddings)


#2. LOAD AND TOKENIZE DATA --------------------------------------------------------------------------------------

#load json dataset
dataset = load_dataset("json", data_files={"train": train_file}, split="train")

print("data loaded")
print(f"length of dataset before filtering for empty:{len(dataset)}")

#remove empty lines
dataset = dataset.filter(is_not_empty)

print(f"length of dataset after filtering for empty:{len(dataset)}")

# total_tokens = sum(len(tokenizer(x["text"])["input_ids"]) for x in dataset)
# print(f"number of tokens in dataset before subsetting: {total_tokens}")


total_tokens = sum(len(tokenizer(x["text"])["input_ids"]) for x in dataset)

#use only a subset of the dataset
if use_subset == True:
    dataset = dataset.select(range(subset_size))

subset_tokens = sum(len(tokenizer(x["text"])["input_ids"]) for x in dataset)
prop_tokens = subset_tokens/total_tokens
print(f"number of tokens in dataset after subsetting: {subset_tokens}")
print(f"proportion of tokens in dataset after subsetting: {prop_tokens}")

#apply tokenizer to the dataset
tokenized_subset = dataset.map(tokenize)

def remove_bad_indices(tokenized_dataset, tokenizer, model):
    bad_indices = []
    model.eval() 
    def is_valid(example, idx):
        try:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            pad_id = tokenizer.pad_token_id
            attention_mask = (input_ids != pad_id).long()

            labels = input_ids.clone()
            labels[input_ids == pad_id] = -100
            # print("Pad token ID:", tokenizer.pad_token_id)
            # print("Max input ID in example:", max(example["input_ids"]))    
            with torch.no_grad():
                # vocab_size = model.get_input_embeddings().num_embeddings
                # print("Model vocab size:", vocab_size)
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

#load and tokenise the validation set
val_dataset_raw = load_dataset("text", data_files={"validation": val_file}, split="validation")
val_dataset = val_dataset_raw.filter(is_not_empty)
tokenized_val = val_dataset.map(tokenize, batched=True)

# print(f"val dataset raw: {val_dataset_raw[0]}")
# print(f"val dataset: {val_dataset[0]}")
# print(f"tokenized val: {tokenized_val[0]}")

tokenized_subset, bad_indices_train = remove_bad_indices(tokenized_subset, tokenizer, model)
tokenized_val, bad_indices_val = remove_bad_indices(tokenized_val, tokenizer, model)

# #print first few examples for debugging
# print(f"Number of training examples: {len(tokenized_subset)}")
# for i, example in enumerate(tokenized_subset):
#     print(f"\n[Example {i}] Input token IDs:\n{example['input_ids']}")
#     print(f"[Example {i}] Decoded text:\n{tokenizer.decode(example['input_ids'], skip_special_tokens=False)}")

#batch sequences and apply padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


#3. REPORT INITIAL LOSS ------------------------------------------------------------------------------------------

#set class for loss logging
# class EpochLossLoggerCallback(TrainerCallback):
#     def on_epoch_end(self, args, state, control, **kwargs):
#         #initialise
#         last_epoch = None
#         train_loss = None
#         eval_loss = None

#         #get most recent losses
#         for log in reversed(state.log_history):
#             if "epoch" in log:
#                 if "loss" in log:
#                     train_loss = log["loss"]
#                     last_epoch = log["epoch"]
#                 if "eval_loss" in log:
#                     eval_loss = log["eval_loss"]
#                     last_epoch = log["epoch"]
#                 if train_loss is not None and eval_loss is not None:
#                     break

#function for computing cross entropy loss
def compute_loss(model, dataset):
    model.eval()
    losses = []
    with torch.no_grad():
        for example in dataset:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            # input_ids = torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
            pad_id = int(tokenizer.pad_token_id)
            attention_mask = (input_ids != pad_id).long()

            labels = input_ids.clone()
            labels[input_ids == tokenizer.pad_token_id] = -100

            if (labels != -100).sum().item() == 0:
                continue

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
# initial_loss, initial_ppl = compute_loss(model, tokenized_subset)
initial_loss, initial_ppl = compute_loss(model, tokenized_val)

print("\n--- Before Training ---")
print(f"Cross-Entropy Loss: {initial_loss:.4f}")
print(f"Perplexity: {initial_ppl:.2f}")



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
    logging_steps="epoch",
    save_total_limit=save_total_limit,
    fp16=False,
    report_to=None,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False #lower loss is better
)

#set up model trainger
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_subset,
    eval_dataset = tokenized_val,
    #eval_dataset = tokenized_subset,
    tokenizer=tokenizer,
    data_collator=data_collator
    # callbacks = [EpochLossLoggerCallback()]
)

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


#6. FINISHING UP --------------------------------------------------------------------------------------------

#report final loss
model.to(device)
#final_loss, final_ppl = compute_loss(model, tokenized_subset)
final_loss, final_ppl = compute_loss(model, tokenized_val)
print("\n--- After Training ---")
print(f"Cross-Entropy Loss: {final_loss:.4f}")
print(f"Perplexity: {final_ppl:.2f}")


#save model 
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training completed.")


#ADD GENERATION FROM HALF PROMPTS ---------------------------------------------------------------------------------------------

#set generation params
max_new_tokens = 30
temperature = 0.8
top_p = 0.9
do_sample = True

#set file paths
finetuned_path = "llm/model_results/llama_finetuned"

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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
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
    "model": model_name,
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accum_steps,
    "num_epochs": num_epochs,
    "max_sequence_length": max_sequence_length,
    "subset_size": subset_size if use_subset else "full dataset",
    "train_tokens": subset_tokens,
    "token_proportion": prop_tokens,
    "best_epoch": best_epoch_num,
    "best_val_loss": best_val_loss,
    "best_loss": final_loss,
    "best_perplexity": final_ppl,
    "bad_indices_train": bad_indices_train,
    "bad_indices_val": bad_indices_val,
    "max_new_tokens": max_new_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "do_sample": do_sample,
    "gaelic_completions": gaelic_completions,
    "english_completions": english_completions
}

metadata_file = os.path.join(results_output_dir, "metadata.json")
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print(f"Saved training metadata to {metadata_file}")
