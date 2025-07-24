import os
import yaml
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from itertools import combinations

def load_metadata(run_folder):
    metadata_path = os.path.join(run_folder, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    else:
        return None


def extract_varying_params_from_grid(grid_params_path):
    # Read all lines (each line is a set of CLI args)
    with open(grid_params_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    # Parse each line into a dict of param: value, supporting both '--param value' and '--param=value'
    param_dicts = []
    for line in lines:
        args = line.split('--')[1:]
        d = {}
        for arg in args:
            arg = arg.strip()
            if '=' in arg:
                k, v = arg.split('=', 1)
                k = k.strip()
                v = v.strip()
            else:
                parts = arg.split(None, 1)
                if len(parts) == 2:
                    k, v = parts
                    k = k.strip()
                    v = v.strip()
                else:
                    continue
            d[k] = v
        param_dicts.append(d)
    # Find params that vary (i.e., have more than one unique value across all lines)
    all_keys = set(k for d in param_dicts for k in d)
    varying = {}
    for k in all_keys:
        values = [d.get(k, None) for d in param_dicts]
        # Remove None values for unique check
        unique_values = set(v for v in values if v is not None)
        if len(unique_values) > 1:
            varying[k] = sorted(unique_values)
    return varying, param_dicts

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

def main(run_id):
    base_dir = os.path.join("llm/finetune/results", f"run_{run_id}")

    grid_params_path = os.path.join(base_dir, "grid_params.txt")
    if not os.path.exists(grid_params_path):
        print(f"No grid_params.txt found in {base_dir}")
        return

    varying_params, param_dicts = extract_varying_params_from_grid(grid_params_path)
    print(f"Identified varying hyperparameters: {list(varying_params.keys())}")

    # Find all logs_* subfolders in this run folder, sorted for consistent mapping
    run_folders = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("logs_")])

    all_data = []
    for i, run in enumerate(run_folders):
        # Get param values from param_dicts (same order as grid_params.txt)
        param_row = {k: param_dicts[i].get(k, None) for k in varying_params.keys()}
        # Add best_loss etc from metadata
        metadata = load_metadata(run)
        if metadata:
            param_row["best_loss"] = metadata.get("best_loss", None)
            param_row["best_val_loss"] = metadata.get("best_val_loss", None)
            param_row["best_epoch"] = metadata.get("best_epoch", None)
            param_row["run_folder"] = os.path.basename(run)
            all_data.append(param_row)

    if not all_data:
        print("No metadata loaded.")
        return


    df = pd.DataFrame(all_data)
    print(df)
    # Only convert param columns to numeric if their type is already numeric, otherwise leave as is
    for param in varying_params:
        # If all values are numeric (int or float as string), convert to numeric
        print(f"Type of param '{param}': {df[param].dtype}, all values: {df[param]}")
        non_null_vals = df[param].dropna().unique()
        if all(isinstance(v, (int, float)) for v in non_null_vals):
            df[param] = pd.to_numeric(df[param])
            print(f"Converted param '{param}' to numeric.")
        else:
            print(f"Param '{param}' left as is (type: {df[param].dtype}, values: {non_null_vals})")

    for param in varying_params:
        if df[param].nunique() > 1:
            plot_param_effects(df, param, base_dir)
            print(f"Saved plot for {param}")



    # --- FacetGrid: For each combination of varying params, plot a facet grid of train/val loss curves ---
    # If only one param varies, do as before. If two, use row/col. If more, iterate over all pairs.
    var_keys = list(varying_params.keys())
    print(F"VAR KEYS: {var_keys}")
    n_var = len(var_keys)
    # Collect all epoch loss data for all runs
    all_records = []
    for run in run_folders:
        epoch_losses_path = os.path.join(run, "epoch_losses.json")
        if not os.path.exists(epoch_losses_path):
            continue
        with open(epoch_losses_path, "r") as f:
            epoch_data = json.load(f)
        train_loss = epoch_data.get("train_loss", [])
        val_loss = epoch_data.get("val_loss", [])
        row = df[df["run_folder"] == os.path.basename(run)]
        if row.empty:
            continue
        record_base = {k: row.iloc[0][k] for k in var_keys}
        for epoch, loss in train_loss:
            rec = {"epoch": epoch, "loss": loss, "set": "train"}
            rec.update(record_base)
            all_records.append(rec)
        for epoch, loss in val_loss:
            rec = {"epoch": epoch, "loss": loss, "set": "val"}
            rec.update(record_base)
            all_records.append(rec)

    if not all_records:
        print("No epoch loss data found.")
        return

    plot_df = pd.DataFrame(all_records)

    # For each varying param, plot a facet grid with columns for that param
    for param in var_keys:
        n_facets = plot_df[param].nunique()
        if n_facets == 0:
            continue
        col_wrap = math.ceil(n_facets ** 0.5) if n_facets > 1 else 1
        g = sns.FacetGrid(plot_df, col=param, hue="set", sharey=True, sharex=True, height=4, aspect=1.2, col_wrap=col_wrap)
        g.map_dataframe(sns.lineplot, x="epoch", y="loss", marker="o")
        g.add_legend()
        g.set_axis_labels("Epoch", "Loss")
        g.set_titles(col_template=f"{param}={{col_name}}")
        plt.subplots_adjust(top=0.85)
        g.fig.suptitle(f"Train/Val Loss by {param}")
        plot_name = f"facetgrid_loss_by_{param}.png"
        g.savefig(os.path.join(base_dir, plot_name))
        plt.close(g.fig)
        print(f"Saved facet grid loss plot for {param} as {plot_name}")

    result_csv_path = os.path.join(base_dir, "gridsearch_results.csv")
    df.to_csv(result_csv_path, index=False)
    print(f"\n Saved full results table to {result_csv_path}")

    print("\nSummary Table:")
    print(df[[*varying_params.keys(), "best_loss", "best_val_loss", "best_epoch", "run_folder"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="Run ID, e.g. 20250722_165137")
    args = parser.parse_args()
    main(args.run_id)
