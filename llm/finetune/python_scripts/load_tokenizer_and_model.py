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
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from peft import PeftModel, PeftConfig
from datasets import load_dataset


def load_tokenizer(device, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(["<|pad|>"])
    tokenizer.pad_token = "<|pad|>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    return tokenizer


# def load_model(tokenizer, device, args):
#     model = AutoModelForCausalLM.from_pretrained(args.model_name)
#     #start with everything frozen
#     for param in model.parameters():
#         param.requires_grad = False
#     #lora
#     if args.peft_mode == "lora" or args.peft_mode == "lora+head":
#         lora_config = LoraConfig(
#             r=args.lora_r,
#             lora_alpha=args.lora_alpha,
#             target_modules=["q_proj", "v_proj"], #TEMP: swap out for args!
#             lora_dropout=args.lora_dropout,
#             bias="none",
#             task_type=TaskType.CAUSAL_LM,
#         )
#         model = get_peft_model(model, lora_config)
#         for name, param in model.named_parameters():
#             param.requires_grad = False

#         for name, param in model.named_parameters():
#             if "lora_" in name:
#                 param.requires_grad = True
#     #add head
#     if args.peft_mode in ["head-only", "lora+head"]:
#         for name, param in model.named_parameters():
#             if "lm_head" in name:
#                 param.requires_grad = True

#     #no parametes frozen if 
#     if args.peft_mode == "none":
#         for param in model.parameters():
#             param.requires_grad = True

    
#     #resize model due to padding tokens in tokenizer
#     model.resize_token_embeddings(len(tokenizer))
#     model.to(device)
#     return model

def load_model(tokenizer, device, args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Resize embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Freeze all parameters by default
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA if needed
    if args.peft_mode in {"lora", "lora+head"}:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],  # TODO: replace with args if configurable
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_bnb_8bit=False,
        )
        model = get_peft_model(model, lora_config)

    # Selectively unfreeze based on peft_mode
    for name, param in model.named_parameters():
        if args.peft_mode == "none":
            param.requires_grad = True
        elif args.peft_mode == "head-only" and "lm_head" in name:
            param.requires_grad = True
        elif args.peft_mode in {"lora", "lora+head"} and "lora_" in name:
            param.requires_grad = True
        elif args.peft_mode == "lora+head" and "lm_head" in name:
            param.requires_grad = True

    return model.to(device)