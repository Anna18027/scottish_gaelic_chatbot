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
    RUN_ID="20250719_190746"
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
SCRATCH_SAVE_DIR="$SCRATCH_FINETUNE_DIR/saved_model" 
SCRATCH_GRID_FILE="$SCRATCH_RUN_DIR/grid_params.txt"

#set output filepaths for home (after copying from scratch)
HOME_DATA_DIR="$HOME_CHATBOT_DIR/data"
HOME_FINETUNE_DIR="$HOME_CHATBOT_DIR/llm/finetune"
HOME_RUN_DIR="$HOME_FINETUNE_DIR/results/run_$RUN_ID"
HOME_SAVE_DIR="$HOME_FINETUNE_DIR/saved_model" 
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

# if $ON_CLUSTER; then
#     #check python version
#     echo "Finding python path"
#     which python

#     echo "Checking python3 version"
#     python3 --version

#     if ! command -v python3.12 &> /dev/null; then
#         echo "Python 3.12 not found, trying conda..."
#         if ! command -v conda &> /dev/null; then
#             echo "Conda is not installed or not in PATH."
#             exit 1
#         fi
#         source $(conda info --base)/etc/profile.d/conda.sh
#         if conda env list | grep -q "py312env"; then
#             echo "Conda env py312env exists, activating..."
#         else
#             echo "Creating conda env py312env with python 3.12..."
#             conda create -n py312env python=3.12 -y
#         fi
#         conda activate py312env
#     else
#         echo "Python 3.12 is installed."
#     fi
#     echo "Checking python3 version"
#     python3 --version
# fi
# if $ON_CLUSTER; then
#     echo "Finding python path"
#     which python

#     echo "Checking python3 version"
#     python3 --version

#     PYTHON_VERSION_INSTALLED=$(python3 --version 2>&1 | awk '{print $2}')
#     if [[ "$PYTHON_VERSION_INSTALLED" != "3.12.0" ]]; then
#         echo "Python 3.12.0 not found, trying conda..."

#         if ! command -v conda &> /dev/null; then
#             echo "Conda is not installed or not in PATH."
#             exit 1
#         fi

#         source $(conda info --base)/etc/profile.d/conda.sh

#         echo "Searching available Python versions in conda:"
#         conda search python

#         if conda env list | grep -q "py312env"; then
#             echo "Conda env py312env exists, activating..."
#         else
#             echo "Creating conda env py312env with python 3.12.0..."
#             conda create -n py312env python=3.12.0 -y
#         fi

#         conda activate py312env
#     else
#         echo "Python 3.12.0 is already installed."
#     fi

#     echo "Final Python version:"
#     python3 --version
# fi
# if $ON_CLUSTER; then
#     echo "Finding python path"
#     which python

#     echo "Checking python3 version"
#     python3 --version

#     PYTHON_VERSION_INSTALLED=$(python3 --version 2>&1 | awk '{print $2}')
#     if [[ "$PYTHON_VERSION_INSTALLED" != "3.12.0" ]]; then
#         echo "Python 3.12.0 not found, trying conda..."

#         if ! command -v conda &> /dev/null; then
#             echo "Conda is not installed or not in PATH."
#             exit 1
#         fi

#         source $(conda info --base)/etc/profile.d/conda.sh
        
#         # echo "Searching available Python versions in conda:"
#         # conda search python

#         if conda env list | grep -q "py312env"; then
#             echo "Conda env py312env exists, activating..."
#         else
#             echo "Creating conda env py312env with python 3.12.0..."
#             conda create -n py312env python=3.12.0 -y
#         fi

#         conda activate py312env
#     else
#         echo "Python 3.12.0 is already installed."
#     fi

#     echo "Final Python version:"
#     python3 --version
# fi
# if $ON_CLUSTER; then
#     echo "Finding python path"
#     which python

#     echo "Checking python3 version"
#     python3 --version

#     PYTHON_VERSION_INSTALLED=$(python3 --version 2>&1 | awk '{print $2}')
#     if [[ "$PYTHON_VERSION_INSTALLED" != "3.12.0" ]]; then
#         echo "Python 3.12.0 not found, trying conda..."

#         if ! command -v conda &> /dev/null; then
#             echo "Conda is not installed or not in PATH."
#             exit 1
#         fi

#         source $(conda info --base)/etc/profile.d/conda.sh
        
#         echo "Available builds for Python 3.12.0:"
#         conda search python=3.12.0 --info

#         if conda env list | grep -q "py312env"; then
#             echo "Conda env py312env exists, activating..."
#         else
#             echo "Creating conda env py312env with python 3.12.0 (build h996f2a0_0)..."
#             conda create -n py312env python=3.12.0=h996f2a0_0 -y
#         fi

#         conda activate py312env
#     else
#         echo "Python 3.12.0 is already installed."
#     fi

#     echo "Final Python version:"
#     python3 --version
# fi
if conda env list | grep -q "^py312env\s"; then
    echo "Conda env py312env exists, checking Python version inside it..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate py312env
    
    PY_VER=$(python --version 2>&1 | awk '{print $2}')
    if [[ "$PY_VER" != "3.12.0" ]]; then
        echo "Python version in py312env is $PY_VER, not 3.12.0. Recreating environment..."
        conda deactivate
        conda env remove -n py312env -y
        conda create -n py312env python=3.12.0 -y
        conda activate py312env
    else
        echo "Python 3.12.0 confirmed in py312env."
    fi
else
    echo "Conda env py312env does not exist. Creating with Python 3.12.0..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda create -n py312env python=3.12.0 -y
    conda activate py312env
fi

echo "Python version before activating venv:"
python3 --version

# #activate venv and install requirements
# if $ON_CLUSTER; then
#     if [ -f "$VENV_PATH/bin/activate" ]; then
#         echo "Activating existing virtual environment..."
#         source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
#         pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
#     else
#         echo "WARNING: Virtual environment not found; creating new one..."
#         python3 -m venv "$VENV_PATH" || { echo "ERROR: Failed to create virtual environment"; exit 1; }
#         source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
#         pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
#     fi
# fi
# Activate venv and install requirements
# if $ON_CLUSTER; then
#     if [ -f "$VENV_PATH/bin/activate" ]; then
#         echo "Virtual environment found. Checking Python version..."
#         source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }

#         VENV_PYTHON_VERSION=$("$VENV_PATH/bin/python3" --version 2>&1 | awk '{print $2}')
#         echo "Python version in venv: $VENV_PYTHON_VERSION"

#         if [[ "$VENV_PYTHON_VERSION" == "3.12.0" ]]; then
#             echo "Python version is 3.12.0 — activating existing virtual environment..."
#             pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
#         else
#             echo "Python version is not 3.12.0 — recreating virtual environment..."
#             deactivate || true
#             rm -rf "$VENV_PATH"
#             python3 -m venv "$VENV_PATH" || { echo "ERROR: Failed to create virtual environment"; exit 1; }
#             source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate new virtual environment"; exit 1; }

#             # Confirm new version
#             echo "New Python version in venv: $(python3 --version)"
#             pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
#         fi
#     else
#         echo "Virtual environment not found — creating new one..."
#         python3 -m venv "$VENV_PATH" || { echo "ERROR: Failed to create virtual environment"; exit 1; }
#         source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }

#         echo "New Python version in venv: $(python3 --version)"
#         pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
#     fi
# fi
if $ON_CLUSTER; then
    if [ -f "$VENV_PATH/bin/activate" ]; then
        echo "Virtual environment found. Checking Python version..."
        source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }

        VENV_PYTHON_VERSION=$("$VENV_PATH/bin/python3" --version 2>&1 | awk '{print $2}')
        echo "Python version in venv: $VENV_PYTHON_VERSION"

        if [[ "$VENV_PYTHON_VERSION" == "3.12.0" ]]; then
            echo "Python version is 3.12.0 — activating existing virtual environment..."
            
            # Fix for missing distutils
            python3 -m ensurepip --upgrade
            pip install --upgrade pip setuptools
            
            pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
        else
            echo "Python version is not 3.12.0 — recreating virtual environment..."
            deactivate || true
            rm -rf "$VENV_PATH"
            python3 -m venv "$VENV_PATH" || { echo "ERROR: Failed to create virtual environment"; exit 1; }
            source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate new virtual environment"; exit 1; }

            echo "New Python version in venv: $(python3 --version)"

            # Fix for missing distutils
            python3 -m ensurepip --upgrade
            pip install --upgrade pip setuptools
            
            pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
        fi
    else
        echo "Virtual environment not found — creating new one..."
        python3 -m venv "$VENV_PATH" || { echo "ERROR: Failed to create virtual environment"; exit 1; }
        source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }

        echo "New Python version in venv: $(python3 --version)"

        # Fix for missing distutils
        python3 -m ensurepip --upgrade
        pip install --upgrade pip setuptools

        pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
    fi
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

    mkdir -p "$LOG_DIR" 

    START_TIME=$(date +%s)

    python3 "$SCRATCH_FINETUNE_DIR/python_scripts/main.py" $PARAM_STRING --run_name "$RUN_NAME" --run_dir "$SCRATCH_RUN_DIR" --save_dir "$SCRATCH_SAVE_DIR" --log_dir "$LOG_DIR"  --train_file "$TRAIN_FILE" --val_file "$VAL_FILE" --model_name "$MODEL_NAME" > "$LOG_FILE" 2>&1
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