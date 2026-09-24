#!/bin/bash
#SBATCH --job-name=byol_galaxy10
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/users/koketso/Feature_extraction/spectra_for_features/logs/byol_galaxy10_%j.log
#SBATCH --error=/users/koketso/Feature_extraction/spectra_for_features/logs/byol_galaxy10_%j.log

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

source /idia/projects/camil/Koketso/.venv/deepclustering3/bin/activate

echo "Job started on $(hostname) at $(date)"
echo "Using environment: $(which python)"

python -u Training_/byol.py

echo "Job finished at $(date)"