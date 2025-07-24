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
    EarlyStoppingCallback,
    AutoConfig
)
from types import SimpleNamespace
from datasets import load_dataset
from bitsandbytes import AdamW8bit

def train_model(model, tokenizer, tokenized_train, tokenized_val, data_collator, args):

    print("bitsandbytes version:")
    print(bnb.__version__)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        optimizers=(AdamW8bit(model.parameters(), lr=args.learning_rate))
    )

    trainer.train()
    return trainer