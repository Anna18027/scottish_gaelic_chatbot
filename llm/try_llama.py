from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Define cache location
user = os.environ["USER"]
cache_dir = f"/exports/eddie/scratch/{user}/hf_models"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer and causal LM model from cache
tokenizer = AutoTokenizer.from_pretrained(cache_dir)
model = AutoModelForCausalLM.from_pretrained(cache_dir)

# Make sure model is on the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model and tokenizer loaded from cache.")

# Input prompt
prompt = "Explain why the sky is blue in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:\n", generated_text)

