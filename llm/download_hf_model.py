from huggingface_hub import login, snapshot_download

# Login with your token (replace 'YOUR_HF_TOKEN' with your actual token)
login(token="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")

# Download the model to the specified directory
snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B",
    local_dir="/exports/eddie/scratch/s2751141/hf_models/models--meta-llama--Llama-3.2-1B",
    local_dir_use_symlinks=False
)

print("Python script complete, model should have downloaded")
