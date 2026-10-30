#!/bin/bash
# Runs all seven intervention strategies against a model served by vLLM's
# OpenAI-compatible server (used for Qwen3-235B-A22B-Thinking-2507 and
# Kimi-K2-Thinking, each on 8× H200). Run from the repo root:
#
#   vllm serve $MODEL --tensor-parallel-size 8 --max-model-len 16384 [--trust-remote-code for Kimi-K2]
#   MODEL=Qwen/Qwen3-235B-A22B-Thinking-2507 DIR=results/intervention/qwen3-235b-a22b-thinking bash intervention/run_vllm.sh
#   MODEL=moonshotai/Kimi-K2-Thinking        DIR=results/intervention/kimi-k2-thinking         bash intervention/run_vllm.sh

# ── Config ──────────────────────────────────────────────────────
MODEL=${MODEL:?set MODEL to the served model name}
DIR=${DIR:?set DIR to the folder holding edits_*.jsonl}
API_URL=${API_URL:-http://localhost:8000/v1}
MAX_LEN=16384
STRATEGIES=(fd fd_branch_comp fd_branch_comp_ctrl bc_minus_d0only bc_minus_d0only_ctrl bc_minus_d0_and_deep1plus bc_minus_d0_and_deep1plus_ctrl)
# ────────────────────────────────────────────────────────────────

for STRAT in "${STRATEGIES[@]}"; do
    EDITS=${DIR}/edits_${STRAT}.jsonl
    OUTPUT=${DIR}/results_${STRAT}.csv

    if [ ! -f "$EDITS" ]; then
        echo "=== SKIP $STRAT: $EDITS not found ==="
        continue
    fi

    echo "=== Running strategy: $STRAT ==="
    python -u intervention/infer_vllm.py \
        --model "$MODEL" \
        --edits "$EDITS" \
        --output "$OUTPUT" \
        --api_url "$API_URL" \
        --max_model_len $MAX_LEN
    echo "=== Done: $STRAT ==="
    echo ""
done

echo "=== All strategies complete ==="
