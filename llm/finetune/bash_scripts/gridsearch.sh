#!/bin/bash
# gridsearch_local.sh

# Set filepaths based on local or cluster run
if [[ "$(hostname)" == *"mlp"* ]]; then
    CHATBOT_DIR="/disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot"
    ON_CLUSTER=true
else
    CHATBOT_DIR="/Users/annamcmanus/Documents/2024-25 Masters Year/Dissertation/scottish_gaelic_chatbot"
    ON_CLUSTER=false
fi

echo "Using CHATBOT_DIR: $CHATBOT_DIR"

# Set relative filepaths
FINETUNE_DIR="$CHATBOT_DIR/llm/finetune"
RUN_DIR="$FINETUNE_DIR/results/run_20250718_140851"
SAVE_DIR="$FINETUNE_DIR/saved_model" 
VENV_PATH="$CHATBOT_DIR/.venv"
REQUIREMENTS_FILE="$CHATBOT_DIR/llm/requirements.txt"
GRID_FILE="$RUN_DIR/grid_params.txt"

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

    python3 "$FINETUNE_DIR/python_scripts/main.py" $PARAM_STRING --run_name "$RUN_NAME" --run_dir "$RUN_DIR" --save_dir "$SAVE_DIR" --log_dir "$LOG_DIR" > "$LOG_FILE" 2>&1
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
