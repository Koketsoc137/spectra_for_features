#!/bin/bash
#SBATCH --job-name=gz_foundation_models
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/gz_foundation_models_%j.log
#SBATCH --error=logs/gz_foundation_models_%j.log

mkdir -p logs

repo_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

source /idia/projects/camil/Koketso/.venv/deepclustering3/bin/activate

echo "Job started on $(hostname) at $(date)"
echo "Using environment: $(which python)"

python -u Training_/gz_foundation_models.py

echo "Job finished at $(date)"
