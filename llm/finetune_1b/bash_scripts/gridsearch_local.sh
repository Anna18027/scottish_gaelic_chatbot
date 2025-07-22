#!/bin/bash
# gridsearch_local.sh

#set filepaths - change between local and cluster
CHATBOT_DIR="/Users/annamcmanus/Documents/2024-25 Masters Year/Dissertation/scottish_gaelic_chatbot"
FINETUNE_DIR="$CHATBOT_DIR/llm/finetune"
RUN_DIR="$FINETUNE_DIR/results/run_20250718_140851"
SAVE_DIR="$FINETUNE_DIR/saved_model" 

#path to grid file (no change between runs)
GRID_FILE="$RUN_DIR/grid_params.txt"

TOTAL_JOBS=$(wc -l < "$GRID_FILE")

for TASK_ID in $(seq 1 $((TOTAL_JOBS))); do
    echo "==== Running local task $TASK_ID ===="

    PARAM_STRING=$(sed -n "$((TASK_ID))p" "$GRID_FILE")
    # RUN_NAME="run_${TASK_ID}"
    LOG_DIR="$RUN_DIR/logs_${TASK_ID}"
    LOG_FILE="$LOG_DIR/output.log"

    mkdir -p "$LOG_DIR" 

    START_TIME=$(date +%s)

    #run training file
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
