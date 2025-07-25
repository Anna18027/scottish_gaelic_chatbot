#!/bin/bash
#SBATCH --job-name=finetune-grid
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=Teach-Standard-Noble
#SBATCH --gres=gpu:gtx_1080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

#set model name
MODEL_NAME="timinar/baby-llama-58m"

#set filepaths based on local or cluster run
if [[ "$PWD" == *"s2751141"* ]]; then
    SCRATCH_CHATBOT_DIR="/disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot"
    HOME_CHATBOT_DIR="/home/s2751141/dissertation/scottish_gaelic_chatbot"
    ON_CLUSTER=true
else
    SCRATCH_CHATBOT_DIR="/Users/annamcmanus/Documents/2024-25 Masters Year/Dissertation/scottish_gaelic_chatbot"
    HOME_CHATBOT_DIR="/Users/annamcmanus/Documents/2024-25 Masters Year/Dissertation/scottish_gaelic_chatbot"
    ON_CLUSTER=false
fi

#set run id
if $ON_CLUSTER; then
    # RUN_ID="20250719_190746"
    RUN_ID="20250725_data"
    # RUN_ID="20250725_lora"
else
    RUN_ID="20250718_140801"
fi

echo "Using HOME_CHATBOT_DIR: $HOME_CHATBOT_DIR"
echo "Using SCRATCH_CHATBOT_DIR: $SCRATCH_CHATBOT_DIR"

#set output filepaths for scratch
VENV_PATH="$SCRATCH_CHATBOT_DIR/.venv"
REQUIREMENTS_FILE="$SCRATCH_CHATBOT_DIR/llm/requirements.txt"
SCRATCH_DATA_DIR="$SCRATCH_CHATBOT_DIR/data"
SCRATCH_FINETUNE_DIR="$SCRATCH_CHATBOT_DIR/llm/finetune"
SCRATCH_RUN_DIR="$SCRATCH_FINETUNE_DIR/results/run_$RUN_ID"
# SCRATCH_SAVE_DIR="$SCRATCH_FINETUNE_DIR/saved_model" 
SCRATCH_GRID_FILE="$SCRATCH_RUN_DIR/grid_params.txt"

#set output filepaths for home (after copying from scratch)
HOME_DATA_DIR="$HOME_CHATBOT_DIR/data"
HOME_FINETUNE_DIR="$HOME_CHATBOT_DIR/llm/finetune"
HOME_RUN_DIR="$HOME_FINETUNE_DIR/results/run_$RUN_ID"
# HOME_SAVE_DIR="$HOME_FINETUNE_DIR/saved_model"
HOME_GRID_FILE="$HOME_RUN_DIR/grid_params.txt"

#copy data across to scratch
if $ON_CLUSTER; then
    mkdir -p "$SCRATCH_DATA_DIR"
    rsync -a "$HOME_DATA_DIR/" "$SCRATCH_DATA_DIR/" || { echo "ERROR: Failed to copy data folder to scratch"; exit 1; }
fi

#set data to use
TRAIN_FILE="$SCRATCH_DATA_DIR/madlad_from_huggingface/gd_clean_0000.jsonl.gz"
VAL_FILE="$SCRATCH_DATA_DIR/temp_data/gaidhlig_test_set.txt"

#copy project folder across to scratch
if $ON_CLUSTER; then
    mkdir -p "$SCRATCH_CHATBOT_DIR/llm"
    rsync -a --delete "$HOME_CHATBOT_DIR/llm/" "$SCRATCH_CHATBOT_DIR/llm" || { echo " ERROR: rsync failed"; exit 1; }
fi

echo "Finding python path"
which python

echo "Checking python3 version"
python3 --version

if $ON_CLUSTER; then
    if ! command -v python3.10 &> /dev/null; then
        echo "Python 3.10 not found, trying conda..."
        if ! command -v conda &> /dev/null; then
            echo "Conda is not installed or not in PATH."
            exit 1
        fi
        source $(conda info --base)/etc/profile.d/conda.sh
        if conda env list | grep -q "py310env"; then
            echo "Conda env py310env exists, activating..."
        else
            echo "Creating conda env py310env with python 3.10..."
            conda create -n py310env python=3.10 -y
        fi
        conda activate py310env
    else
        echo "Python 3.10 is installed."
    fi
fi


if $ON_CLUSTER; then
    if [ -f "$VENV_PATH/bin/activate" ]; then
        VENV_PYTHON_VERSION=$("$VENV_PATH/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [[ "$VENV_PYTHON_VERSION" == "3.10" ]]; then
            echo "Activating existing Python 3.10 virtual environment..."
            source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
        else
            echo "Existing virtual environment Python version is $VENV_PYTHON_VERSION, but Python 3.10 required."
            echo "Deleting old venv and creating new one with Python 3.10..."
            rm -rf "$VENV_PATH"
            python3.10 -m venv "$VENV_PATH" || { echo "ERROR: Failed to create virtual environment"; exit 1; }
            source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
        fi
    else
        echo "No virtual environment found; creating new one with Python 3.10..."
        python3.10 -m venv "$VENV_PATH" || { echo "ERROR: Failed to create virtual environment"; exit 1; }
        source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
    fi

    pip cache purge
    python -m pip install --upgrade pip setuptools wheel || { echo "ERROR: Failed to upgrade pip/setuptools"; exit 1; }
    pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
fi

echo "Python version after activating venv:"
python3 --version


echo "progress check 2"

#get total number of tasks
TOTAL_JOBS=$(wc -l < "$SCRATCH_GRID_FILE")

#run python script for each task
for TASK_ID in $(seq 1 $((TOTAL_JOBS))); do
    echo "==== Running task $TASK_ID of $TOTAL_JOBS ===="


    PARAM_STRING=$(sed -n "$((TASK_ID))p" "$SCRATCH_GRID_FILE")
    LOG_DIR="$SCRATCH_RUN_DIR/logs_$TASK_ID"
    LOG_FILE="$LOG_DIR/output.log"
    SAVE_DIR="$LOG_DIR/saved_model"

    mkdir -p "$LOG_DIR"

    START_TIME=$(date +%s)

    python3 "$SCRATCH_FINETUNE_DIR/python_scripts/main.py" $PARAM_STRING --run_name "$RUN_NAME" --run_dir "$SCRATCH_RUN_DIR" --save_dir "$SAVE_DIR" --log_dir "$LOG_DIR"  --train_file "$TRAIN_FILE" --val_file "$VAL_FILE" --model_name "$MODEL_NAME" > "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [[ $EXIT_CODE -ne 0 ]] || grep -q "Traceback" "$LOG_FILE"; then
        echo "Task $TASK_ID/$TOTAL_JOBS FAILED (in ${DURATION}s)"
        echo "Check log: $LOG_FILE"
    else
        echo "Task $TASK_ID/$TOTAL_JOBS completed successfully in $DURATION seconds"
    fi

    #copy outputs back to home
    if $ON_CLUSTER; then
        if [ -d "$SCRATCH_RUN_DIR" ]; then
            rsync -av "$SCRATCH_RUN_DIR/" "$HOME_RUN_DIR/"
        else
            echo "ERROR: Results not found in $SCRATCH_RUN_DIR"
            exit 1
        fi
    fi
done

echo "bash script complete"