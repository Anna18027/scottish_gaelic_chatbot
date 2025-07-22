print("starting python script")

from transformers import AutoModel, AutoTokenizer
import os

print("library imports complete")

# Define the model name
#model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "meta-llama/Llama-3.2-1B"

# Optional: Set a custom cache directory in your home dir
user = os.environ["USER"]
# cache_dir = f"/exports/eddie/scratch/{user}/hf_models"
#cache_dir = "/home/$USER/hf_models"
cache_dir = f"/exports/eddie/scratch/{user}/hf_models"

# Download the tokenizer and model
print(f"Downloading {model_name} to {cache_dir}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

tokenizer.save_pretrained(cache_dir)
model.save_pretrained(cache_dir)

print("Download complete.")
