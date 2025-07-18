import os
import yaml
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def find_run_folders(base_dir):
    return sorted(
        [os.path.join(base_dir, d) for d in os.listdir(base_dir)
         if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("cfg_")]
    )

def load_metadata(run_folder):
    metadata_path = os.path.join(run_folder, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    else:
        return None

def extract_varying_params(config):
    return {
        k: v for k, v in config.items()
        if isinstance(v, list) and len(v) > 1
    }

def flatten_run_data(metadata, varying_keys):
    result = {k: metadata.get(k, None) for k in varying_keys}
    result["best_loss"] = metadata.get("best_loss", None)
    result["best_val_loss"] = metadata.get("best_val_loss", None)
    result["best_epoch"] = metadata.get("best_epoch", None)
    return result

def plot_param_effects(df, param_name, output_dir):
    plt.figure()
    grouped = df.groupby(param_name).mean(numeric_only=True)
    grouped[["best_loss", "best_val_loss"]].plot(marker='o')
    plt.title(f"Effect of {param_name}")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{param_name}_effect.png"))
    plt.close()

def main(base_dir):

    base_dir = os.path.join("llm/finetune/results", base_dir)

    config_path = os.path.join(base_dir, "used_config.yaml")
    if not os.path.exists(config_path):
        print(f"No used_config.yaml found in {base_dir}")
        return

    config = load_config(config_path)
    varying_params = extract_varying_params(config)

    print(f"Identified varying hyperparameters: {list(varying_params.keys())}")

    run_folders = find_run_folders(base_dir)
    if not run_folders:
        print("No run folders found.")
        return

    all_data = []
    for run in run_folders:
        metadata = load_metadata(run)
        if metadata:
            row = flatten_run_data(metadata, varying_params.keys())
            row["run_folder"] = os.path.basename(run)
            all_data.append(row)

    if not all_data:
        print("No metadata loaded.")
        return

    df = pd.DataFrame(all_data)

    for param in varying_params:
        if df[param].nunique() > 1:
            plot_param_effects(df, param, base_dir)
            print(f"Saved plot for {param}")

    result_csv_path = os.path.join(base_dir, "gridsearch_results.csv")
    df.to_csv(result_csv_path, index=False)
    print(f"\n Saved full results table to {result_csv_path}")

    print("\nSummary Table:")
    print(df[[*varying_params.keys(), "best_loss", "best_val_loss", "best_epoch", "run_folder"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to the folder containing run_* subfolders")
    args = parser.parse_args()
    main(args.folder)
