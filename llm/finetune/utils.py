import yaml
import os

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def is_running_on_cluster():
    if os.path.exists("/home/s2751141"):
        return True
    elif os.path.exists("/Users/annamcmanus"):
        return False
    else:
        raise EnvironmentError(
            "Neither teaching cluster nor local environment detected. Check code logic."
        )

def is_grid_search(config_dict):
    return any(isinstance(v, list) and len(v) > 1 for v in config_dict.values())

def get_run_mode(config):
    running_on_cluster = is_running_on_cluster()
    grid_search = is_grid_search(config)

    if grid_search and running_on_cluster:
        mode = "grid_search_cluster"
    elif grid_search and not running_on_cluster:
        mode = "grid_search_local"
    else:
        mode = "interactive"
    return mode
