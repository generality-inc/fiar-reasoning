#!/bin/bash
#SBATCH --job-name=intervention
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=./results/intervention.log

# ── Config ──────────────────────────────────────────────────────
# Set MODEL to the local path of the Qwen3-Next-80B-A3B-Thinking weights.
MODEL=${MODEL:-/path/to/Qwen3-Next-80B-A3B-Thinking}
TP=4
MAX_LEN=16384
EDITS_DIR=results/figure4_intervention
OUTPUT_DIR=results/figure4_intervention
STRATEGIES=(fd fd_branch_comp fd_branch_comp_ctrl bc_minus_d0only bc_minus_d0only_ctrl bc_minus_d0_and_deep1plus bc_minus_d0_and_deep1plus_ctrl)
# ────────────────────────────────────────────────────────────────

mkdir -p $OUTPUT_DIR

for STRAT in "${STRATEGIES[@]}"; do
    EDITS=${EDITS_DIR}/intervention_edits_${STRAT}.jsonl
    OUTPUT=${OUTPUT_DIR}/intervention_results_${STRAT}.csv

    if [ ! -f "$EDITS" ]; then
        echo "=== SKIP $STRAT: $EDITS not found ==="
        continue
    fi

    echo "=== Running strategy: $STRAT ==="
    python -u intervention_infer.py \
        --model $MODEL \
        --edits $EDITS \
        --output $OUTPUT \
        --tensor_parallel_size $TP \
        --max_model_len $MAX_LEN
    echo "=== Done: $STRAT ==="
    echo ""
done

echo "=== All figure4 strategies complete ==="
