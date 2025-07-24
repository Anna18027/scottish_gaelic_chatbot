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


def load_model(tokenizer, device, args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    #full finetuning as default
    for param in model.parameters():
        param.requires_grad = True
    #lora
    if args.peft_mode == "lora" or args.peft_mode == "lora+head":
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"], #TEMP: swap out for args!
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    #add head
    if args.peft_mode in ["head-only", "lora+head"]:
        for name, param in model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True
    #resize model due to padding tokens in tokenizer
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return model