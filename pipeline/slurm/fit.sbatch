#!/bin/bash
# One cognitive-model fit per LLM (array index -> fiar.models.FITTED_MODELS).
# Run from the repo root:
#   VARIANT=myopic sbatch pipeline/slurm/fit.sbatch
#   VARIANT=gamma BASIS=windowcount sbatch pipeline/slurm/fit.sbatch   # after the windowcount myopic fits
#SBATCH --job-name=fiar-fit
#SBATCH --array=0-12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fit-%A_%a.out

mkdir -p logs
python -u pipeline/fit_cognitive_models.py \
    --jobid "$SLURM_ARRAY_TASK_ID" --variant "${VARIANT:-full}" --basis "${BASIS:-pattern}"
