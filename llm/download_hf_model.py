from huggingface_hub import login, snapshot_download, whoami, model_info, logout

logout()
# Login with your token (replace 'YOUR_HF_TOKEN' with your actual token)
login(token="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")

print(whoami())

try:
    info = model_info("meta-llama/Llama-3.2-1B", token="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")
    print("✅ You have access to the model.")
    print("Model files:", list(info.siblings))
except Exception as e:
    print("❌ Access denied or model not found.")
    print("Error:", e)

try:
    info = model_info("meta-llama/Llama-3.2-3B", token="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")
    print("✅ You have access to the model.")
    print("Model files:", list(info.siblings))
except Exception as e:
    print("❌ Access denied or model not found.")
    print("Error:", e)


# # Download the model to the specified directory
# snapshot_download(
#     repo_id="meta-llama/Llama-3.2-1B",
#     # local_dir="/exports/eddie/scratch/s2751141/hf_models/models--meta-llama--Llama-3.2-1B",
#     # local_dir="/disk/scratch/s2751141/hf_models/models--meta-llama--Llama-3.2-1B",
#     local_dir="/home/s2751141/dissertation/scottish_gaelic_chatbot/hf_models/models--meta-llama--Llama-3.2-1B",
#     local_dir_use_symlinks=False
# )

# Download the model to the specified directory
try:
    # snapshot_download(
    #     repo_id="meta-llama/Llama-3.2-3B",
    #     # local_dir="/home/s2751141/dissertation/scottish_gaelic_chatbot/hf_models/models--meta-llama--Llama-3.2-3B",
    #     local_dir="/disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot/hf_models/models--meta-llama--Llama-3.2-3B",
    #     local_dir_use_symlinks=False
    # )
    snapshot_download(
        repo_id="meta-llama/Llama-3.2-1B",
        local_dir="/home/s2751141/dissertation/scottish_gaelic_chatbot/hf_models/models--meta-llama--Llama-3.2-1B",
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.json",               # config and tokenizer
            "*.model",              # tokenizer.model (if SentencePiece)
            "*.safetensors",        # model weights (safetensors)
            "*.safetensors.index.json",  # sharded weight index
            "*.safetensors*",
            "tokenizer.*",          # tokenizer files
            "generation_config.json",
            "special_tokens_map.json"
        ],
        resume_download=True,
        token = "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"
    )
except Exception as e:
    print(f"Error during snapshot_download: {e}")

print("Python script complete, 1B model should have downloaded")
