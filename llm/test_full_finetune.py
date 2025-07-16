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
from datetime import datetime
import os

#set model params
num_epochs = 1
# model_name = "meta-llama/Llama-3.2-1B"  #"timinar/baby-llama-58m" #"meta-llama/Llama-3.2-1B"
use_subset = True
subset_size = 20
batch_size = 1
gradient_accum_steps = 1 #8
logging_steps = 1
save_total_limit = 1

#set file paths
train_file = "/disk/scratch/s2751141/data/madlad_from_huggingface/gd_clean_0000.jsonl.gz"
val_file = "/disk/scratch/s2751141/data/temp_data/gaidhlig_test_set.txt"
output_dir = "/disk/scratch/s2751141/model_results/llama_finetuned"
model_path = "/disk/scratch/s2751141/hf_models/models--meta-llama--Llama-3.2-3B"
# model_path = "hf_models/llama_3_2_1B"
# model_path = "hf_models/models--meta-llama--Llama-3.2-1B"

#NEW CODE
use_wandb = False

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#1. LOAD TOKENIZER & MODEL --------------------------------------------------------------------------------------

print("Current working directory:", os.getcwd())
# print("Contents of /disk/scratch/s2751141/hf_models:")
# print(os.listdir("/disk/scratch/s2751141/hf_models"))

print(f"Checking if model path exists: {model_path}")
if os.path.exists(model_path):
    print("Model path exists.")
    print("Contents of model directory:")
    print(os.listdir(model_path))
else:
    print("Model path does NOT exist!")

print("starting to load model")

# Load model

try:
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        use_auth_token=False
    )
    print("Model loaded successfully. First try.")
except Exception as e:
    print("Error loading model (first try):")
    print(e)

try:
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="/disk/scratch_big/s2751141/hf_models/models--meta-llama--Llama-3.2-3B",
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        use_auth_token=False
    )
    print("Model loaded successfully. Second try.")
except Exception as e:
    print("Error loading model (second try):")
    print(e)

print("model loaded")
