#!/bin/bash
# Runs all Figure-4 intervention strategies against a model served by vLLM's
# OpenAI-compatible server (used for Qwen3-235B-A22B-Thinking-2507 and Kimi-K2-Thinking).
#
#   vllm serve $MODEL --tensor-parallel-size 8 --max-model-len 16384
#   MODEL=Qwen/Qwen3-235B-A22B-Thinking-2507 DIR=results/intervention_qwen3-235b-a22b-thinking bash run_intervention_api.sh
#   MODEL=moonshotai/Kimi-K2-Thinking        DIR=results/intervention_kimi-k2-thinking        bash run_intervention_api.sh

# ── Config ──────────────────────────────────────────────────────
MODEL=${MODEL:?set MODEL to the served model name}
DIR=${DIR:?set DIR to the folder holding intervention_edits_*.jsonl}
API_URL=${API_URL:-http://localhost:8000/v1}
MAX_LEN=16384
STRATEGIES=(fd fd_branch_comp fd_branch_comp_ctrl bc_minus_d0only bc_minus_d0only_ctrl bc_minus_d0_and_deep1plus bc_minus_d0_and_deep1plus_ctrl)
# ────────────────────────────────────────────────────────────────

for STRAT in "${STRATEGIES[@]}"; do
    EDITS=${DIR}/intervention_edits_${STRAT}.jsonl
    OUTPUT=${DIR}/intervention_results_${STRAT}.csv

    if [ ! -f "$EDITS" ]; then
        echo "=== SKIP $STRAT: $EDITS not found ==="
        continue
    fi

    echo "=== Running strategy: $STRAT ==="
    python -u intervention_infer_api.py \
        --model $MODEL \
        --edits $EDITS \
        --output $OUTPUT \
        --api_url $API_URL \
        --max_model_len $MAX_LEN
    echo "=== Done: $STRAT ==="
    echo ""
done

echo "=== All strategies complete ==="
