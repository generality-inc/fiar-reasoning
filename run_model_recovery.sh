#!/bin/bash
#SBATCH --job-name=model_recovery
#SBATCH --array=0-13          # one job per open-source model (14 total)
#SBATCH --time=1:00:00
#SBATCH --mem=5G
#SBATCH --cpus-per-task=1
#SBATCH --output=./results/recovery_%a.log

python -u run_model_recovery.py \
    --jobid $SLURM_ARRAY_TASK_ID \
    --n_restarts 5 \
    --pkl game_trees_df_annotated.pkl

# After all array jobs complete, aggregate (run manually or add a dependent job):
# python run_model_recovery.py --all  (skips already-done models, just aggregates)
