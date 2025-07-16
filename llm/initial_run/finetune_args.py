import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Baby Llama model")

    #model name
    parser.add_argument("--model_name", type=str, default="timinar/baby-llama-58m", help="Name of huggingface model")

    #data size
    parser.add_argument("--subset_size", type=int, default=20, help="Subset size of training data")

    #filepaths
    parser.add_argument("--train_file", type=str, default="llm/data/madlad_from_huggingface/gd_clean_0000.jsonl.gz", help="Path to training data")
    parser.add_argument("--val_file", type=str, default="llm/data/temp_data/gaidhlig_test_set.txt", help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="llm/model_results/llama_finetuned", help="Directory to save models and results")

    #model parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--max_sequence_length", type=int, default=256, help="Max token length per sequence")
    parser.add_argument("--gradient_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Save total limit")

    #PEFT
    parser.add_argument("--peft", type=str, choices=["none", "head only", "LORA only", "LORA and head"], default="none", help="PEFT strategy")

    #generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling probability")
    parser.add_argument("--do_sample", type=bool, default=True, help="Whether to use sampling for generation")

    args = parser.parse_args()
    return args
