#!/bin/bash
#SBATCH --job-name=finetune-llm
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# mkdir -p /disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot/llm

# # Sync source contents into destination
# rsync -a --delete "/home/s2751141/dissertation/scottish_gaelic_chatbot/llm/" \
#                 "/disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot/llm/" \
#   || { echo " ERROR: rsync failed"; exit 1; }


#move working directory to scratch for venv activation
cd /home/s2751141/dissertation/scottish_gaelic_chatbot/ || { echo "ERROR: Failed to cd to scratch directory"; exit 1; }

#activate venv
if [ -f ".venv/bin/activate" ]; then
    echo "Activating existing virtual environment..."
    source .venv/bin/activate
    pip install -r llm/requirements.txt || { echo "ERROR: Failed to install requirements"; exit 1; }
else
    echo "WARNING: Virtual environment not found in scratch; creating new one..."
    python3 -m venv .venv || { echo "ERROR: Failed to create virtual environment"; exit 1; }
    source .venv/bin/activate || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
    pip install -r llm/requirements.txt || { echo "ERROR: Failed to install requirements"; exit 1; }
fi

#run python file on home
python /home/s2751141/dissertation/scottish_gaelic_chatbot/llm/download_hf_model.py || { echo "Python script failed with exit code $?"; exit 1; }


# #sync outputs back from scratch to disk
# rsync -av disk/scratch/s2751141/model_results /home/s2751141/dissertation/scottish_gaelic_chatbot/model_results

