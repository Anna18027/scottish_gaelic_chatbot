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

def kwargs_to_args(kwargs):
    return SimpleNamespace(**kwargs)

def setup_environment(args):
    start_time = datetime.now()
    timestamp = args.timestamp
    os.makedirs(args.full_output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"0. Initial set-up is complete. Total time taken: {get_elapsed_time(start_time)}")
    return start_time, timestamp, args.full_output_dir, device

def load_model_and_tokenizer(device, start_time, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(["<|pad|>"])
    tokenizer.pad_token = "<|pad|>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    print(f"1. Load tokenizer and model is complete. Total time taken:{get_elapsed_time(start_time)}")
    return tokenizer, model

def load_and_preprocess_data(tokenizer, model, device, start_time, args):
    # Load train data
    dataset = load_dataset("json", data_files={"train": args.train_file}, split="train")
    print("data loaded")
    print(f"length of dataset before filtering for empty:{len(dataset)}")
    dataset = dataset.filter(is_not_empty)
    print(f"length of dataset after filtering for empty:{len(dataset)}")

    texts = list(dataset["text"])
    batch_encoding = tokenizer(texts, truncation=False, padding=False)
    total_tokens = sum(len(ids) for ids in batch_encoding["input_ids"])

    if args.subset_size > 0:
        dataset = dataset.select(range(args.subset_size))

    texts = list(dataset["text"])
    batch_encoding = tokenizer(texts, truncation=False, padding=False)
    subset_tokens = sum(len(ids) for ids in batch_encoding["input_ids"])
    prop_tokens = subset_tokens / total_tokens

    print(f"number of tokens in dataset after subsetting: {subset_tokens}")
    print(f"proportion of tokens in dataset after subsetting: {prop_tokens}")

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            max_length=args.max_sequence_length,
        )

    tokenized_subset = dataset.map(tokenize, batched=True, batch_size=1000)

    val_dataset_raw = load_dataset("text", data_files={"validation": args.val_file}, split="validation")
    val_dataset = val_dataset_raw.filter(is_not_empty)
    tokenized_val = val_dataset.map(tokenize, batched=True, batch_size=1000)

    tokenized_subset, bad_indices_train = remove_bad_indices(tokenized_subset, tokenizer, model, device)
    tokenized_val, bad_indices_val = remove_bad_indices(tokenized_val, tokenizer, model, device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print(f"2. Load and tokenize data is complete. Total time taken:{get_elapsed_time(start_time)}")

    return tokenized_subset, tokenized_val, data_collator, subset_tokens, prop_tokens, bad_indices_train, bad_indices_val

def report_initial_loss(model, tokenized_val, tokenizer, device, start_time):
    initial_loss, initial_ppl = compute_loss(model, tokenized_val, tokenizer, device)
    print("\n--- Before Training ---")
    print(f"Cross-Entropy Loss: {initial_loss:.4f}")
    print(f"Perplexity: {initial_ppl:.2f}")
    print(f"3. Report initial loss is complete. Total time taken:{get_elapsed_time(start_time)}")

def train_model(model, tokenizer, tokenized_subset, tokenized_val, data_collator, start_time, args):
    training_args = TrainingArguments(
        output_dir=args.saved_model_dir,
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
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_subset,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print(f"4. Model training is complete. Total time taken:{get_elapsed_time(start_time)}")
    return trainer

def save_losses_and_plot(trainer, output_dir, start_time):
    epoch_train_losses = []
    epoch_val_losses = []

    for log in trainer.state.log_history:
        if "epoch" in log:
            epoch = log["epoch"]
            if "loss" in log:
                epoch_train_losses.append((epoch, log["loss"]))
            if "eval_loss" in log:
                epoch_val_losses.append((epoch, log["eval_loss"]))

    loss_file = os.path.join(output_dir, "epoch_losses.json")
    with open(loss_file, "w") as f:
        json.dump(
            {
                "train_loss": epoch_train_losses,
                "val_loss": epoch_val_losses,
            },
            f,
        )

    best_epoch_idx = min(range(len(epoch_val_losses)), key=lambda i: epoch_val_losses[i][1])
    best_epoch_num, best_val_loss = epoch_val_losses[best_epoch_idx]

    print(f"Best Epoch: {best_epoch_num:.1f} with Validation Loss: {best_val_loss:.4f}")

    train_epochs = [x[0] for x in epoch_train_losses]
    train_losses = [x[1] for x in epoch_train_losses]

    val_epochs = [x[0] for x in epoch_val_losses]
    val_losses = [x[1] for x in epoch_val_losses]

    plt.figure(figsize=(6, 4))
    plt.plot(train_epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(val_epochs, val_losses, marker="s", label="Validation Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    print(f"Saved loss curve to {output_dir}/loss_curve.png")
    print(f"Saved raw loss values to {loss_file}")
    print(f"5. Save losses per epoch is complete. Total time taken:{get_elapsed_time(start_time)}")

    return best_epoch_num, best_val_loss

def finishing_up(model, tokenizer, trainer, tokenized_val, device, output_dir, start_time, args):
    model.to(device)
    final_loss, final_ppl = compute_loss(model, tokenized_val, tokenizer, device)
    print("\n--- After Training ---")
    print(f"Cross-Entropy Loss: {final_loss:.4f}")
    print(f"Perplexity: {final_ppl:.2f}")

    trainer.save_model(args.saved_model_dir)
    tokenizer.save_pretrained(args.saved_model_dir)
    print("Training completed.")

    return final_loss, final_ppl

def generate_samples(finetuned_path, device, start_time, timestamp, output_dir,
                     best_epoch_num, best_val_loss, final_loss, final_ppl,
                     bad_indices_train, bad_indices_val, args):
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_path)
    finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
    finetuned_model.to(device)
    finetuned_model.eval()

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

        gaelic_completions.append({"prompt": gaelic_prompt, "completion": completion_gaelic})
        english_completions.append({"prompt": english_prompt, "completion": completion_english})

    metadata = {
        "timestamp": timestamp,
        "model": args.model_name,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accum_steps,
        "num_epochs": args.num_epochs,
        "max_sequence_length": args.max_sequence_length,
        "subset_size": args.subset_size if args.subset_size > 0 else "full dataset",
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
        "english_completions": english_completions,
    }

    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"Saved training metadata to {metadata_file}")
    print(f"6. Finishing up and generation is complete. Total time taken: {get_elapsed_time(start_time)}")


def run_training(**kwargs):
    args = kwargs_to_args(kwargs)

    start_time, timestamp, output_dir, device = setup_environment(args)
    tokenizer, model = load_model_and_tokenizer(device=device, start_time=start_time, args=args)
    (
        tokenized_subset,
        tokenized_val,
        data_collator,
        subset_tokens,
        prop_tokens,
        bad_indices_train,
        bad_indices_val,
    ) = load_and_preprocess_data(tokenizer=tokenizer, model=model, device=device, start_time=start_time, args=args)

    report_initial_loss(model, tokenized_val, tokenizer, device, start_time)
    trainer = train_model( model=model, tokenizer=tokenizer, tokenized_subset=tokenized_subset, tokenized_val=tokenized_val, data_collator=data_collator, start_time=start_time, args=args)
    best_epoch_num, best_val_loss = save_losses_and_plot(trainer, output_dir, start_time)
    final_loss, final_ppl = finishing_up(model, tokenizer, trainer, tokenized_val, device, output_dir, start_time, args=args)

    generate_samples(
        finetuned_path=args.saved_model_dir,
        device=device,
        start_time=start_time,
        timestamp=timestamp,
        output_dir=args.full_output_dir,
        best_epoch_num=best_epoch_num,
        best_val_loss=best_val_loss,
        final_loss=final_loss,
        final_ppl=final_ppl,
        bad_indices_train=bad_indices_train,
        bad_indices_val=bad_indices_val,
        args=args
    )

