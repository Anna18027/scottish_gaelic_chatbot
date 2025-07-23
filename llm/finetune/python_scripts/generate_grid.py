import itertools
import os
from datetime import datetime

cwd = os.getcwd()

if "s2751141" in cwd:
    print("Cluster version (mlp)")
    chatbot_dir = "/home/s2751141/dissertation/scottish_gaelic_chatbot"
elif "studios" in cwd:
    print("Lightning version")
    chatbot_dir = 'scottish_gaelic_chatbot'
else:
    print("Local version")
    chatbot_dir = "/Users/annamcmanus/Documents/2024-25 Masters Year/Dissertation/scottish_gaelic_chatbot"

base_dir = os.path.join(chatbot_dir, "llm", "finetune", "results")

print("Using base_dir:", base_dir)

# === 1. Define All Parameters Here ===

params = {
    # Data subset
    "subset_size": [20, 0],

    # Training
    "batch_size": 16,
    "learning_rate": 3e-5,
    "num_epochs": 50,
    "max_sequence_length": 256,
    "gradient_accum_steps": 8,
    "save_total_limit": 1,
    "early_stopping_patience": 3,

    # PEFT
    "peft_mode": "none",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_target_modules": "q_proj",

    # Generation
    "max_new_tokens": 30,
    "temperature": 0.8,
    "top_p": 0.9,
    "do_sample": True,
}

# === 2. Split Params Into Fixed and Grid ===

fixed_params = {k: v for k, v in params.items() if not isinstance(v, list)}
grid_params = {k: v for k, v in params.items() if isinstance(v, list) and len(v) > 1}

# === 3. Generate Cartesian Product ===

keys = list(grid_params.keys())
combinations = list(itertools.product(*[grid_params[k] for k in keys]))

print(f"Generating grid for {len(combinations)} combinations...")

# === 4. Output Directory ===

run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
run_dir = os.path.join(base_dir, run_id)
os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

grid_file_path = os.path.join(run_dir, "grid_params.txt")

with open(grid_file_path, "w") as grid_file:
    for idx, combo in enumerate(combinations):
        full_params = dict(zip(keys, combo)) | fixed_params  # Merge combo + fixed
        args_string = " ".join([f"--{k} {v}" for k, v in full_params.items()])
        grid_file.write(args_string + "\n")

        # # Save config file
        # cfg_path = os.path.join(run_id, f"cfg_{idx}.txt")
        # with open(cfg_path, "w") as cfg:
        #     cfg.write(args_string + "\n")

print(f"Saved {len(combinations)} grid entries to: {grid_file_path}")
