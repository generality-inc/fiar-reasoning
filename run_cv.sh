#!/bin/bash
#SBATCH --job-name=4iar
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0-26

mkdir -p ./results

python -u run_cv.py \
    --jobid=$SLURM_ARRAY_TASK_ID