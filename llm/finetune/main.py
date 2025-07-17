import yaml
from datetime import datetime
from copy import deepcopy
from itertools import product
import os
from utils import load_config, get_run_mode
from train import run_training

#load config file
config_path = "llm/finetune/config.yaml"
config = load_config(config_path)

#set up run mode
mode = get_run_mode(config)

#create output folder
run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
base_output_dir = os.path.join(config["output_dir"], run_id)
os.makedirs(base_output_dir, exist_ok=True)
timestamp = run_id[len("run_"):]

#run training script (looped if needed)
if mode == "interactive":
    full_output_dir = os.path.join(base_output_dir, "run_1")
    os.makedirs(full_output_dir, exist_ok=True)

    interactive_config = deepcopy(config)
    interactive_config["full_output_dir"] = full_output_dir
    interactive_config["timestamp"] = timestamp

    run_training(**interactive_config)

elif mode == "grid_search_local":
    print(f"Running in {mode} mode")
    
    grid_params = {k: v for k, v in config.items() if isinstance(v, list) and len(v) > 1}
    static_params = {k: v for k, v in config.items() if not (isinstance(v, list) and len(v) > 1)}

    keys = list(grid_params.keys())
    values = list(grid_params.values())

    for idx, combo in enumerate(product(*values)):
        combo_config = deepcopy(static_params)
        combo_dict = dict(zip(keys, combo))
        combo_config.update(combo_dict)

        # Short run name
        run_name = f"cfg_{idx:02d}"

        # Full output dir = base dir + run name
        full_output_dir = os.path.join(base_output_dir, run_name)
        os.makedirs(full_output_dir, exist_ok=True)

        # Pass it into config
        combo_config["full_output_dir"] = full_output_dir
        combo_config["timestamp"] = timestamp

        print(f"Running config: {run_name}")
        run_training(**combo_config)
elif mode == "grid_search_cluster":
    print(f"mode is {mode}")
else: 
    raise ValueError(f"Invalid mode: {mode}")

