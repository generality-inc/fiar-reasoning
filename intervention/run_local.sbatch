#!/bin/bash
#SBATCH --job-name=intervention
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#
# Runs all seven intervention strategies for Qwen3-Next-80B-A3B-Thinking with
# local vLLM. Submit from the repo root:
#
#   MODEL_DIR=/path/to/Qwen3-Next-80B-A3B-Thinking sbatch intervention/run_local.sbatch
#
# If MODEL_DIR does not exist yet, the weights are downloaded there first.

# ── Config ──────────────────────────────────────────────────────
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the Qwen3-Next-80B-A3B-Thinking weights folder}
DIR=${DIR:-results/intervention/qwen3-next-80b-a3b-thinking}
TP=4
MAX_LEN=16384
STRATEGIES=(fd fd_branch_comp fd_branch_comp_ctrl bc_minus_d0only bc_minus_d0only_ctrl bc_minus_d0_and_deep1plus bc_minus_d0_and_deep1plus_ctrl)
# ────────────────────────────────────────────────────────────────

if [ ! -d "$MODEL_DIR" ]; then
    echo "=== Downloading Qwen/Qwen3-Next-80B-A3B-Thinking to $MODEL_DIR ==="
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3-Next-80B-A3B-Thinking \
        --local-dir "$MODEL_DIR"
fi

for STRAT in "${STRATEGIES[@]}"; do
    EDITS=${DIR}/edits_${STRAT}.jsonl
    OUTPUT=${DIR}/results_${STRAT}.csv

    if [ ! -f "$EDITS" ]; then
        echo "=== SKIP $STRAT: $EDITS not found ==="
        continue
    fi

    echo "=== Running strategy: $STRAT ==="
    python -u intervention/infer_local.py \
        --model "$MODEL_DIR" \
        --edits "$EDITS" \
        --output "$OUTPUT" \
        --tensor_parallel_size $TP \
        --max_model_len $MAX_LEN
    echo "=== Done: $STRAT ==="
    echo ""
done

echo "=== All strategies complete ==="
