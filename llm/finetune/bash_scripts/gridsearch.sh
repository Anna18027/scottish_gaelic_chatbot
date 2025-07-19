#!/bin/bash
#SBATCH --job-name=finetune-llm
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teach-Standard-Noble
#SBATCH --gres=gpu:gtx_1080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Set filepaths based on local or cluster run
if [[ "$(hostname)" == *"mlp"* ]]; then
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

#set filepaths for home
HOME_FINETUNE_DIR="$HOME_CHATBOT_DIR/llm/finetune"
RUN_DIR="$HOME_FINETUNE_DIR/results/run_20250718_140851" #home
# RUN_DIR="$HOME_FINETUNE_DIR/results/run_20250719_190746" #cluster
SAVE_DIR="$HOME_FINETUNE_DIR/saved_model" 
GRID_FILE="$RUN_DIR/grid_params.txt"

#set filepaths for scratch
SCRATCH_FINETUNE_DIR="$SCRATCH_CHATBOT_DIR/llm/finetune"
VENV_PATH="$SCRATCH_CHATBOT_DIR/.venv"
REQUIREMENTS_FILE="$SCRATCH_CHATBOT_DIR/llm/requirements.txt"

echo "progress check"


# Only activate or create venv if on cluster
if $ON_CLUSTER; then
    if [ -f "$VENV_PATH/bin/activate" ]; then
        echo "Activating existing virtual environment..."
        source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
        pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
    else
        echo "WARNING: Virtual environment not found; creating new one..."
        python3 -m venv "$VENV_PATH" || { echo "ERROR: Failed to create virtual environment"; exit 1; }
        source "$VENV_PATH/bin/activate" || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
        pip install -r "$REQUIREMENTS_FILE" || { echo "ERROR: Failed to install requirements"; exit 1; }
    fi
fi

echo "progress check 2"

# Check total number of tasks
TOTAL_JOBS=$(wc -l < "$GRID_FILE")

# Run python script for each task
for TASK_ID in $(seq 1 $((TOTAL_JOBS))); do
    echo "==== Running task $TASK_ID of $TOTAL_JOBS ===="

    PARAM_STRING=$(sed -n "$((TASK_ID))p" "$GRID_FILE")
    LOG_DIR="$RUN_DIR/logs_${TASK_ID}"
    LOG_FILE="$LOG_DIR/output.log"

    mkdir -p "$LOG_DIR" 

    START_TIME=$(date +%s)

    python3 "$SCRATCH_FINETUNE_DIR/python_scripts/main.py" $PARAM_STRING --run_name "$RUN_NAME" --run_dir "$RUN_DIR" --save_dir "$SAVE_DIR" --log_dir "$LOG_DIR" > "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [[ $EXIT_CODE -ne 0 ]] || grep -q "Traceback" "$LOG_FILE"; then
        echo "Task $TASK_ID/$TOTAL_JOBS FAILED (in ${DURATION}s)"
        echo "Check log: $LOG_FILE"
    else
        echo "Task $TASK_ID/$TOTAL_JOBS completed successfully in $DURATION seconds"
    fi
done
