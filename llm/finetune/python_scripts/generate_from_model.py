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
    AutoConfig,
)
from peft import PeftModel, PeftConfig


def generate_samples(best_epoch_num, best_val_loss, final_loss, final_ppl,
                     bad_indices_train, bad_indices_val, device, args):
    # finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_path)
    # finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
    finetuned_tokenizer = AutoTokenizer.from_pretrained(args.save_dir)

    print(f"Save dir is {args.save_dir}")

    if finetuned_tokenizer.pad_token is None:
        finetuned_tokenizer.add_tokens(["<|pad|>"])
        finetuned_tokenizer.pad_token = "<|pad|>"
        finetuned_tokenizer.pad_token_id = finetuned_tokenizer.convert_tokens_to_ids("<|pad|>")

    if args.peft_mode != "none":
        peft_config = PeftConfig.from_pretrained(args.save_dir)
        base_config = AutoConfig.from_pretrained(peft_config.base_model_name_or_path)
        base_config.vocab_size = len(finetuned_tokenizer)
        base_model = AutoModelForCausalLM.from_config(base_config)
        base_model.resize_token_embeddings(len(finetuned_tokenizer))
        finetuned_model = PeftModel.from_pretrained(base_model, args.save_dir)
    else:
        # finetuned_model = AutoModelForCausalLM.from_pretrained(args.save_dir)
        # finetuned_model.resize_token_embeddings(len(finetuned_tokenizer))
        config = AutoConfig.from_pretrained(args.save_dir)
        config.vocab_size = len(finetuned_tokenizer)
        finetuned_model = AutoModelForCausalLM.from_config(config)
        finetuned_model.resize_token_embeddings(len(finetuned_tokenizer))
        finetuned_model.load_state_dict(torch.load(os.path.join(args.save_dir, "pytorch_model.bin"), map_location=device))

    # Now ready to generate
    finetuned_model.eval()
    finetuned_model.to(device)

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
        "model": args.model_name,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accum_steps,
        "num_epochs": args.num_epochs,
        "max_sequence_length": args.max_sequence_length,
        "subset_size": args.subset_size if args.subset_size > 0 else "full dataset",
        "peft_mode": args.peft_mode,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": ["q_proj", "v_proj"], #args.lora_target_modules,
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

    metadata_file = os.path.join(args.log_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"Saved training metadata to {metadata_file}")

