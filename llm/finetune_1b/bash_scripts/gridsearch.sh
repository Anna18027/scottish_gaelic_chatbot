#!/bin/bash
#SBATCH --job-name=b1-finetune-grid
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teach-Standard-Noble
#SBATCH --gres=gpu:gtx_1080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

#set run id
# RUN_ID="20250723_125548"
RUN_ID="20250722_125318"

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

echo "Using HOME_CHATBOT_DIR: $HOME_CHATBOT_DIR"
echo "Using SCRATCH_CHATBOT_DIR: $SCRATCH_CHATBOT_DIR"

#set output filepaths for scratch
VENV_PATH="$SCRATCH_CHATBOT_DIR/.venv"
REQUIREMENTS_FILE="$SCRATCH_CHATBOT_DIR/llm/requirements.txt"
SCRATCH_DATA_DIR="$SCRATCH_CHATBOT_DIR/data"
SCRATCH_FINETUNE_DIR="$SCRATCH_CHATBOT_DIR/llm/finetune_1b"
SCRATCH_RUN_DIR="$SCRATCH_FINETUNE_DIR/results/run_$RUN_ID"
SCRATCH_SAVE_DIR="$SCRATCH_FINETUNE_DIR/saved_model" 
SCRATCH_GRID_FILE="$SCRATCH_RUN_DIR/grid_params.txt"

#set output filepaths for home (after copying from scratch)
HOME_DATA_DIR="$HOME_CHATBOT_DIR/data"
HOME_FINETUNE_DIR="$HOME_CHATBOT_DIR/llm/finetune_1b"
HOME_RUN_DIR="$HOME_FINETUNE_DIR/results/run_$RUN_ID"
HOME_SAVE_DIR="$HOME_FINETUNE_DIR/saved_model" 
HOME_GRID_FILE="$HOME_RUN_DIR/grid_params.txt"

#set model name and download folder
MODEL_NAME="meta-llama/Llama-3.2-1B"
MODEL_DIR_NAME="models--meta-llama--Llama-3.2-1B" #set to "none" for timinar
HOME_MODEL_DOWNLOAD_DIR="$HOME_CHATBOT_DIR/hf_models/$MODEL_DIR_NAME" 
SCRATCH_MODEL_DOWNLOAD_DIR="$SCRATCH_CHATBOT_DIR/hf_models/$MODEL_DIR_NAME" 

#copy data across to scratch
if $ON_CLUSTER; then
    mkdir -p "$SCRATCH_DATA_DIR"
    rsync -a "$HOME_DATA_DIR/" "$SCRATCH_DATA_DIR/" || { echo "ERROR: Failed to copy data folder to scratch"; exit 1; }
fi

#copy downloaded model across to scratch
if $ON_CLUSTER && [ -n "$MODEL_DIR_NAME" ] && [ "$MODEL_DIR_NAME" != "none" ]; then
    mkdir -p "$SCRATCH_MODEL_DOWNLOAD_DIR"
    rsync -a "$HOME_MODEL_DOWNLOAD_DIR/" "$SCRATCH_MODEL_DOWNLOAD_DIR/" || { echo "ERROR: Failed to copy model folder to scratch"; exit 1; }
else
    echo "Skipping model copy: not on cluster or MODEL_DIR_NAME is unset/None"
fi

#set data to use
TRAIN_FILE="$SCRATCH_DATA_DIR/madlad_from_huggingface/gd_clean_0000.jsonl.gz"
VAL_FILE="$SCRATCH_DATA_DIR/temp_data/gaidhlig_test_set.txt"

#copy project folder across to scratch
if $ON_CLUSTER; then
    mkdir -p "$SCRATCH_CHATBOT_DIR/llm"
    rsync -a --delete "$HOME_CHATBOT_DIR/llm/" "$SCRATCH_CHATBOT_DIR/llm" || { echo " ERROR: rsync failed"; exit 1; }
fi

# echo "about to load python 3.10"
# #load pythom 3.10
# if ! command -v module &> /dev/null; then
#     source /etc/profile.d/modules.sh
# fi

# module load python/3.10.4 || { echo "ERROR: Failed to load python/3.10.4 module"; exit 1; }
# echo "after loading"

#activate venv and install requirements
# if $ON_CLUSTER; then
#     if [ -f "$VENV_PATH/bin/activate" ]; then
#         VENV_PYTHON_VERSION=$("$VENV_PATH/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

#         if [[ "$VENV_PYTHON_VERSION" == "3.10" ]]; then
#             echo "Activating existing Python 3.10 virtual environment..."
#             source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
#         else
#             echo "Existing venv uses Python $VENV_PYTHON_VERSION — recreating with Python 3.10..."
#             rm -rf "$VENV_PATH"
#             python3.10 -m venv "$VENV_PATH" || { echo "ERROR: Failed to create virtual environment"; exit 1; }
#             source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate new virtual environment"; exit 1; }
#         fi
#     else
#         echo "No virtual environment found — creating with Python 3.10..."
#         python3.10 -m venv "$VENV_PATH" || { echo "ERROR: Failed to create virtual environment"; exit 1; }
#         source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate new virtual environment"; exit 1; }
#     fi

#     # Install requirements either way
#     pip install --upgrade pip setuptools || { echo "ERROR: Failed to upgrade pip/setuptools"; exit 1; }
#     pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
# fi

echo "Finding python path"
which python

echo "Checking python3 version"
python3 --version

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

# if $ON_CLUSTER; then
#     if [ -f "$VENV_PATH/bin/activate" ]; then
#         echo "Activating existing virtual environment..."
#         source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
#         pip cache purge
#         python -m pip install --upgrade pip setuptools wheel || { echo "ERROR: Failed to upgrade pip/setuptools"; exit 1; }
#         pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
#     else
#         echo "WARNING: Virtual environment not found; creating new one..."
#         pip install python3.10
#         python3.10 -m venv "$VENV_PATH" || { echo "ERROR: Failed to create virtual environment"; exit 1; }
#         source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
#         pip cache purge
#         python -m pip install --upgrade pip setuptools wheel || { echo "ERROR: Failed to upgrade pip/setuptools"; exit 1; }
#         pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
#     fi
# fi

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

echo "progress check 2"

#cd to scratch - best practice for running python script
if $ON_CLUSTER; then
    echo "Changing working directory to scratch chatbot directory..."
    cd "$SCRATCH_CHATBOT_DIR" || { echo "ERROR: Failed to cd to $SCRATCH_CHATBOT_DIR"; exit 1; }
    echo "Current working directory after cd: $(pwd)"
fi


#get total number of tasks
TOTAL_JOBS=$(wc -l < "$SCRATCH_GRID_FILE")

#run python script for each task
for TASK_ID in $(seq 1 $((TOTAL_JOBS))); do
    echo "==== Running task $TASK_ID of $TOTAL_JOBS ===="

    PARAM_STRING=$(sed -n "$((TASK_ID))p" "$SCRATCH_GRID_FILE")
    LOG_DIR="$SCRATCH_RUN_DIR/logs_$TASK_ID"
    LOG_FILE="$LOG_DIR/output.log"

    mkdir -p "$LOG_DIR" 

    START_TIME=$(date +%s)

    python3 "$SCRATCH_FINETUNE_DIR/python_scripts/main.py" $PARAM_STRING --run_name "$RUN_NAME" --run_dir "$SCRATCH_RUN_DIR" --save_dir "$SCRATCH_SAVE_DIR" --log_dir "$LOG_DIR"  --train_file "$TRAIN_FILE" --val_file "$VAL_FILE" --model_name "$MODEL_NAME" --model_download_dir "$SCRATCH_MODEL_DOWNLOAD_DIR" > "$LOG_FILE" 2>&1
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

echo "progress check 3"

echo "bash script complete"