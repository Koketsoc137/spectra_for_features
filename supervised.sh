#!/bin/bash
#SBATCH --job-name=supervised_galaxy10
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/users/koketso/Feature_extraction/spectra_for_features/logs/supervised_galaxy10_%j.log
#SBATCH --error=/users/koketso/Feature_extraction/spectra_for_features/logs/supervised_galaxy10_%j.log

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$repo_dir"

source /idia/projects/camil/Koketso/.venv/deepclustering3/bin/activate

echo "Job started on $(hostname) at $(date)"
echo "Using environment: $(which python)"

python -u Training_/supervised.py

echo "Job finished at $(date)"