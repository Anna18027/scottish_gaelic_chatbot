#!/bin/bash
# gridsearch_local.sh

#set filepaths - change between local and cluster
CHATBOT_DIR="/Users/annamcmanus/Documents/2024-25 Masters Year/Dissertation/scottish_gaelic_chatbot"
FINETUNE_DIR="$CHATBOT_DIR/llm/finetune"
RUN_DIR="$FINETUNE_DIR/results/run_20250718_140851"
SAVE_DIR="$FINETUNE_DIR/saved_model" 

#path to grid file (no change between runs)
GRID_FILE="$RUN_DIR/grid_params.txt"

# echo "Chatbot dir is $CHATBOT_DIR"


# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" #up one directory from this bash script
# echo "Script directory is $SCRIPT_DIR"

# RUN_DIR="$SCRIPT_DIR/results/run_20250718_140851"
# GRID_FILE="$RUN_DIR/grid_params.txt"
# # LOG_DIR="$RUN_DIR/logs"
# SAVE_DIR="$SCRIPT_DIR/saved_model"

# mkdir -p "$LOG_DIR"

TOTAL_JOBS=$(wc -l < "$GRID_FILE")

for TASK_ID in $(seq 1 $((TOTAL_JOBS))); do
    echo "==== Running local task $TASK_ID ===="

    PARAM_STRING=$(sed -n "$((TASK_ID))p" "$GRID_FILE")
    # RUN_NAME="run_${TASK_ID}"
    LOG_DIR="$RUN_DIR/logs_${TASK_ID}"
    LOG_FILE="$LOG_DIR/output.log"

    mkdir -p "$LOG_DIR" 

    echo "log dir is $LOG_DIR"

    # echo "[$RUN_NAME] Params: $PARAM_STRING"
    # echo "Started at $(date)" > "$LOG_FILE"

    START_TIME=$(date +%s)

    #run training file
    python3 llm/finetune/python_scripts/main.py $PARAM_STRING --run_name "$RUN_NAME" --run_dir "$RUN_DIR" --save_dir "$SAVE_DIR" --log_dir "$LOG_DIR" > "$LOG_FILE" 2>&1

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "==== Task $TASK_ID completed in $DURATION seconds ===="

    # echo "Finished at $(date)" >> "$LOG_FILE"
    # echo ""
done
