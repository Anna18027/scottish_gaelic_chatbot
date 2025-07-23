import argparse
import torch
import os
from utils import load_config, get_run_mode, is_running_on_cluster
from train_functions import compute_loss
from load_tokenizer_and_model import load_tokenizer, load_model
from process_data import load_data, tokenize_data, process_data
from train_model import train_model
from evaluate_model import save_losses_and_plot, finishing_up
from generate_from_model import generate_samples

# Check if running on cluster
cluster_running = is_running_on_cluster()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Main script for data loading and model training")

    # Run metadata
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--timestamp', type=str, default="TEMP")

    #model name
    parser.add_argument('--model_name', type=str, default="timinar/baby-llama-58m")
    parser.add_argument('--model_download_dir', type=str, default="none")

    # File paths
    parser.add_argument('--train_file', type=str, default="llm/data/madlad_from_huggingface/gd_clean_0000.jsonl.gz", help="Path to the training data file")
    parser.add_argument('--val_file', type=str, default="llm/data/temp_data/gaidhlig_test_set.txt", help="Path to the validation data file")

    # Training data & schedule
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--subset_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--gradient_accum_steps", type=int, default=8)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    # PEFT settings
    parser.add_argument("--peft_mode", type=str, default="none", choices=["none", "lora", "head-only", "lora+head"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", type=eval, default=True)

    return parser.parse_args()


def main():

    print("Current working directory:", os.getcwd())

    args = parse_arguments()
    kwargs = vars(args)

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_per_process_memory_fraction(1.0)
    torch.backends.cuda.max_split_size_mb = 64

    #load and tokenize data
    train_dataset, val_dataset = load_data(args)
    tokenizer = load_tokenizer(device, args)
    tokenized_train, tokenized_val = tokenize_data(train_dataset, val_dataset, tokenizer, args)

    #load model
    model = load_model(tokenizer, device, args)

    print("MODEL LOADED!!!")

    #further data processing (and subsetting)
    tokenized_train, tokenized_val, prop_tokens, data_collator = process_data(tokenized_train, tokenized_val, tokenizer, model, device, args)

    #compute initial loss
    initial_loss, initial_ppl = compute_loss(model, tokenized_val, tokenizer, device)

    #train model
    trainer = train_model(model, tokenizer, tokenized_train, tokenized_val, data_collator, args)

    #save model results
    best_epoch_num, best_val_loss = save_losses_and_plot(trainer, args)
    final_loss, final_ppl = finishing_up(model, tokenizer, trainer, tokenized_val, device, args)

    #generate from prompts
    # generate_samples()


if __name__ == "__main__":
    main()
