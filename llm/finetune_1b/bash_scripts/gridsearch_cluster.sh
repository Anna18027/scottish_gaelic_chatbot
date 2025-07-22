#!/bin/bash
#SBATCH --job-name=finetune-llm
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teach-Standard-Noble
#SBATCH --gres=gpu:gtx_1080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

CHATBOT_DIR="/disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot"
VENV_PATH="$CHATBOT_DIR/.venv"
REQUIREMENTS_FILE="$CHATBOT_DIR/llm/requirements.txt"

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


#run python file on scratch
python "$CHATBOT_DIR/llm/finetune/main.py" || { echo "Python script failed with exit code $?"; exit 1; }




#!/bin/bash
# gridsearch_local.sh

#set filepaths based on local or cluster run
if [[ "$(hostname)" == *"mlp"* ]]; then
    CHATBOT_DIR="/disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot"
else
    CHATBOT_DIR="/Users/annamcmanus/Documents/2024-25 Masters Year/Dissertation/scottish_gaelic_chatbot"
fi

echo "Using CHATBOT_DIR: $CHATBOT_DIR"

#set relative filepaths
FINETUNE_DIR="$CHATBOT_DIR/llm/finetune"
RUN_DIR="$FINETUNE_DIR/results/run_20250718_140851"
SAVE_DIR="$FINETUNE_DIR/saved_model" 
VENV_PATH="$CHATBOT_DIR/.venv"
REQUIREMENTS_FILE="$CHATBOT_DIR/llm/requirements.txt"
GRID_FILE="$RUN_DIR/grid_params.txt"

#activate venv and install requirements
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

#check total number of tasks
TOTAL_JOBS=$(wc -l < "$GRID_FILE")

#run python script for each task
for TASK_ID in $(seq 1 $((TOTAL_JOBS))); do
    echo "==== Running local task $TASK_ID ===="

    PARAM_STRING=$(sed -n "$((TASK_ID))p" "$GRID_FILE")
    LOG_DIR="$RUN_DIR/logs_${TASK_ID}"
    LOG_FILE="$LOG_DIR/output.log"

    mkdir -p "$LOG_DIR" 

    START_TIME=$(date +%s)

    python3 "$FINETUNE_DIR/python_scripts/main.py" $PARAM_STRING --run_name "$RUN_NAME" --run_dir "$RUN_DIR" --save_dir "$SAVE_DIR" --log_dir "$LOG_DIR" > "$LOG_FILE" 2>&1

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    #echo "==== Task $TASK_ID/$TOTAL_JOBS completed in $DURATION seconds ===="
    if [[ $EXIT_CODE -ne 0 ]] || grep -q "Traceback" "$LOG_FILE"; then
        echo "Task $TASK_ID/$TOTAL_JOBS FAILED (in ${DURATION}s)"
        echo "Check log: $LOG_FILE"
    else
        echo "Task $TASK_ID/$TOTAL_JOBS completed successfully in $DURATION seconds"
    fi
done
