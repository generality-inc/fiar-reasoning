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
# ────────────────────────────────────────────────────────────────

# ── Branch intervention ─────────────────────────────────────────
BRANCH_DIR=results/branch_intervention
BRANCH_STRATEGIES=(fd fd_branch fd_comp fd_branch_ctrl fd_comp_ctrl fd_branch_comp fd_branch_comp_ctrl)

mkdir -p $BRANCH_DIR

for STRAT in "${BRANCH_STRATEGIES[@]}"; do
    EDITS=${BRANCH_DIR}/intervention_edits_${STRAT}.jsonl
    OUTPUT=${BRANCH_DIR}/intervention_results_${STRAT}.csv

    if [ ! -f "$EDITS" ]; then
        echo "=== SKIP $STRAT: $EDITS not found ==="
        continue
    fi

    echo "=== Running branch strategy: $STRAT ==="
    python -u intervention_infer.py \
        --model $MODEL \
        --edits $EDITS \
        --output $OUTPUT \
        --tensor_parallel_size $TP \
        --max_model_len $MAX_LEN
    echo "=== Done: $STRAT ==="
    echo ""
done

# ── Depth intervention ──────────────────────────────────────────
DEPTH_DIR=results/depth_intervention
DEPTH_STRATEGIES=(fd_d0only fd_d0only_ctrl fd_deep1plus fd_deep1plus_ctrl fd_d0_and_deep fd_d0_and_deep_ctrl fd_deep fd_deep_ctrl bc_minus_d0only bc_minus_d0only_ctrl bc_minus_deep1plus bc_minus_deep1plus_ctrl bc_minus_d0_and_deep1plus bc_minus_d0_and_deep1plus_ctrl)

mkdir -p $DEPTH_DIR

for STRAT in "${DEPTH_STRATEGIES[@]}"; do
    EDITS=${DEPTH_DIR}/intervention_edits_${STRAT}.jsonl
    OUTPUT=${DEPTH_DIR}/intervention_results_${STRAT}.csv

    if [ ! -f "$EDITS" ]; then
        echo "=== SKIP $STRAT: $EDITS not found ==="
        continue
    fi

    echo "=== Running depth strategy: $STRAT ==="
    python -u intervention_infer.py \
        --model $MODEL \
        --edits $EDITS \
        --output $OUTPUT \
        --tensor_parallel_size $TP \
        --max_model_len $MAX_LEN
    echo "=== Done: $STRAT ==="
    echo ""
done

echo "=== All intervention strategies complete ==="
