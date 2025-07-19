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
from train_functions import compute_loss


def save_losses_and_plot(trainer, args):
    epoch_train_losses = []
    epoch_val_losses = []

    for log in trainer.state.log_history:
        if "epoch" in log:
            epoch = log["epoch"]
            if "loss" in log:
                epoch_train_losses.append((epoch, log["loss"]))
            if "eval_loss" in log:
                epoch_val_losses.append((epoch, log["eval_loss"]))

    loss_file = os.path.join(args.log_dir, "epoch_losses.json")
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
    plt.savefig(os.path.join(args.log_dir, "loss_curve.png"))
    plt.close()

    print(f"Saved loss curve to {args.log_dir}/loss_curve.png")
    print(f"Saved raw loss values to {loss_file}")

    return best_epoch_num, best_val_loss


def finishing_up(model, tokenizer, trainer, tokenized_val, device, args):
    model.to(device)
    final_loss, final_ppl = compute_loss(model, tokenized_val, tokenizer, device)
    print("\n--- After Training ---")
    print(f"Cross-Entropy Loss: {final_loss:.4f}")
    print(f"Perplexity: {final_ppl:.2f}")

    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print("Training completed.")

    return final_loss, final_ppl