#!/bin/bash
#SBATCH --job-name=finetune-llm
#SBATCH --output=test_logs/finetune-%j.out
#SBATCH --error=test_logs/finetune-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teach-Standard-Noble
#SBATCH --gres=gpu:gtx_1080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

echo "Hello"

#set filepaths
SCRATCH_CHATBOT_DIR="/disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot"
HOME_CHATBOT_DIR="/home/s2751141/dissertation/scottish_gaelic_chatbot"
VENV_PATH="$SCRATCH_CHATBOT_DIR/.venv"
REQUIREMENTS_FILE="$SCRATCH_CHATBOT_DIR/llm/requirements.txt"

HOME_DATA_DIR="/home/s2751141/dissertation/scottish_gaelic_chatbot/data"
HOME_DATA_FILE="$HOME_DATA_DIR/temp_data/english_test_set.txt"
SCRATCH_DATA_DIR="$SCRATCH_CHATBOT_DIR/data"

# OUTPUT_DIR="$SCRATCH_CHATBOT_DIR/test_results"
OUTPUT_DIR="$HOME_CHATBOT_DIR/test_results"

#copy project folder across to scratch
mkdir -p "$SCRATCH_CHATBOT_DIR/llm"
rsync -a --delete "$HOME_CHATBOT_DIR/llm/" \
                "$SCRATCH_CHATBOT_DIR/llm" \
  || { echo " ERROR: rsync failed"; exit 1; }

#activate venv in scratch
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

echo "Before data copied"

#copy data across to scratch
mkdir -p "$SCRATCH_DATA_DIR"
cp "$HOME_DATA_FILE" "$SCRATCH_DATA_DIR"
echo "Data copied"

#run python file from scratch
python "$SCRATCH_CHATBOT_DIR/llm/finetune/python_scripts/test_gridsearch.py" --output_dir $OUTPUT_DIR|| { echo "Python script failed with exit code $?"; exit 1; }

#debugging
echo "SCRATCH dir: $SCRATCH_CHATBOT_DIR"
hostname
ls -l "$SCRATCH_CHATBOT_DIR"

#copy outputs back to home
if [ -d "$SCRATCH_CHATBOT_DIR/test_results" ]; then
    rsync -av "$SCRATCH_CHATBOT_DIR/test_results/" "$HOME_CHATBOT_DIR/test_results/"
else
    echo "ERROR: test_results not found in $SCRATCH_CHATBOT_DIR"
    exit 1
fi

