#!/bin/bash
# Model recovery per LLM (array index -> fiar.models.FITTED_MODELS); needs the pattern full and myopic fits.
# Run from the repo root: sbatch pipeline/slurm/recovery.sbatch
#SBATCH --job-name=fiar-recovery
#SBATCH --array=0-12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=12:00:00
#SBATCH --output=logs/recovery-%A_%a.out

mkdir -p logs
python -u pipeline/model_recovery.py --jobid "$SLURM_ARRAY_TASK_ID"
