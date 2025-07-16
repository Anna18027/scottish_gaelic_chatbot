#!/bin/bash
#SBATCH --job-name=finetune-llm
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


export LLM=/home/s2751141/dissertation/scottish_gaelic_chatbot/llm
export CHATBOT=/home/s2751141/dissertation/scottish_gaelic_chatbot 
export DISS=/home/s2751141/dissertation 
export SCRATCH=/disk/scratch/s2751141
export LOGS=/home/s2751141/dissertation/scottish_gaelic_chatbot/llm/logs

echo "SCRATCH is: $SCRATCH"
echo "LLM is: $LLM"
echo "CHATBOT is: $CHATBOT"
echo "DISS is: $DISS"
echo "LOGS is: $LOGS"

export HF_HOME=/$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=/$SCRATCH/hf_cache/transformers
export HF_DATASETS_CACHE=/$SCRATCH/hf_cache/datasets
export HUGGINGFACE_HUB_TOKEN="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"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

echo "HF_HOME is: $HF_HOME"
echo "TRANSFORMERS_CACHE is: $TRANSFORMERS_CACHE"
echo "HF_DATASETS_CACHE is: $HF_DATASETS_CACHE"
echo "PYTORCH_CUDA_ALLOC_CONF is: $PYTORCH_CUDA_ALLOC_CONF"
echo "CUDA_LAUNCH_BLOCKING is: $CUDA_LAUNCH_BLOCKING"

mkdir -p /disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot/llm

# Sync source contents into destination
rsync -a --delete "/home/s2751141/dissertation/scottish_gaelic_chatbot/llm/" \
                "/disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot/llm/" \
  || { echo " ERROR: rsync failed"; exit 1; }

# #test permissions
# touch /disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot/llm/testfile || echo "No write access to scratch"

# #copy llm project folder to scratch
# rsync -a --delete "/home/s2751141/dissertation/scottish_gaelic_chatbot/llm/" "/disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot/llm/" || { echo "ERROR: rsync failed"; exit 1; }

#move working directory to scratch for venv activation
cd /disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot/ || { echo "ERROR: Failed to cd to scratch directory"; exit 1; }

# if [ -f ".venv/bin/activate" ]; then
#     source .venv/bin/activate
# else
#     echo "WARNING: Virtual environment not found in scratch; please create it or sync it first."
#     python3 -m venv .venv
#     source .venv/bin/activate
#     pip install -r llm/requirements.txt
# fi

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

#run python file on scratch
# python /disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot/llm/full_finetune.py || { echo "Python script failed with exit code $?"; exit 1; }
python /disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot/llm/test_full_finetune.py || { echo "Python script failed with exit code $?"; exit 1; }

#sync outputs back from scratch to disk
rsync -av disk/scratch/s2751141/model_results /home/s2751141/dissertation/scottish_gaelic_chatbot/model_results

