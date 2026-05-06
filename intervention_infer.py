"""
intervention_infer.py

Given edited reasoning traces from intervention_edit.py, run the local model
with edited traces as prefill (temperature=0) and compare to original moves.

Usage:
    python intervention_infer.py \
        --model /path/to/Qwen3-235B-A22B-FP8 \
        --edits results/intervention_edits.jsonl \
        --output results/intervention_results.csv \
        --tensor_parallel_size 4
"""

import argparse
import json
import re
import csv
import os
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_move(text):
    m = re.search(r'<next_move>(.*?)</next_move>', text or '', re.DOTALL)
    return m.group(1).strip() if m else None


def move_to_coord(move_str):
    """Normalize any move format to 'r,c' string, or None if unparseable."""
    if not move_str:
        return None
    m = re.match(r'^\s*m\s*(\d+)\s+(\d+)\s*$', move_str, re.IGNORECASE)
    if m:
        return f"{m.group(1)},{m.group(2)}"
    m = re.match(r'^\s*(\d+)\s*,\s*(\d+)\s*$', move_str)
    if m:
        return f"{m.group(1)},{m.group(2)}"
    return None


def load_edits(edits_path):
    turns = []
    with open(edits_path) as f:
        for line in f:
            turns.append(json.loads(line))
    print(f'Loaded {len(turns)} edited turns from {edits_path}')
    return turns


def build_prompts(turns, tokenizer, max_model_len=16384, max_new_tokens=64):
    """Build prefill prompts using the EDITED reasoning traces."""
    all_prompts = []
    all_meta = []
    skipped = 0
    max_input_len = max_model_len - max_new_tokens

    for turn in turns:
        messages = turn['messages']
        edited_reasoning = turn['edited_reasoning']

        try:
            base = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            base = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if '<think>' not in base:
            base = base + '<think>\n'

        prompt = base + edited_reasoning + '\n</think>\n'

        token_len = len(tokenizer.encode(prompt))
        if token_len > max_input_len:
            skipped += 1
            continue

        all_prompts.append(prompt)
        all_meta.append({
            'turn_idx_global': turn['turn_idx_global'],
            'file': turn['file'],
            'original_move': turn['original_move'],
            'chosen_coord': turn['chosen_coord'],
            'pruned_root': turn['pruned_root'],
            'prune_strategy': turn['prune_strategy'],
            'n_root_moves': turn['n_root_moves'],
            'tree_roots': turn['tree_roots'],
        })

    print(f'Skipped {skipped} turns exceeding {max_input_len} tokens '
          f'({len(all_prompts)} remaining)')
    return all_prompts, all_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--edits', required=True,
                        help='JSONL file from intervention_edit.py')
    parser.add_argument('--output', default='results/intervention_results.csv')
    parser.add_argument('--tensor_parallel_size', type=int, default=4)
    parser.add_argument('--max_model_len', type=int, default=16384)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    t_start = time.time()

    turns = load_edits(args.edits)

    print(f'Loading model from {args.model}...')
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
        stop=['<|im_end|>'],
    )

    all_prompts, all_meta = build_prompts(
        turns, tokenizer, max_model_len=args.max_model_len,
    )
    print(f'Running {len(all_prompts)} inference calls...')

    outputs = llm.generate(all_prompts, sampling_params)

    rows = []
    for meta, output in zip(all_meta, outputs):
        generated = output.outputs[0].text
        new_move = parse_move(generated) or generated.strip()[:20]
        new_coord = move_to_coord(new_move)

        original_coord = meta['chosen_coord']
        pruned_root = meta['pruned_root']
        remaining_roots = [r for r in meta['tree_roots'] if r != pruned_root]

        # Did the model stop playing the pruned move?
        pruned_suppressed = int(new_coord != pruned_root) if new_coord else int(new_move != meta['original_move'])
        # Among remaining roots (not pruned), did the model pick one?
        in_remaining = int(new_coord in remaining_roots) if new_coord and remaining_roots else 0

        rows.append({
            'turn_idx_global': meta['turn_idx_global'],
            'file': meta['file'],
            'prune_strategy': meta['prune_strategy'],
            'n_root_moves': meta['n_root_moves'],
            'pruned_root': pruned_root,
            'tree_roots': '|'.join(meta['tree_roots']),
            'original_move': meta['original_move'],
            'original_coord': original_coord,
            'new_move': new_move,
            'new_coord': new_coord,
            'move_changed': pruned_suppressed,
            'new_move_was_in_remaining_roots': in_remaining,
        })

    fieldnames = [
        'turn_idx_global', 'file', 'prune_strategy', 'n_root_moves',
        'pruned_root', 'tree_roots',
        'original_move', 'original_coord', 'new_move', 'new_coord',
        'move_changed', 'new_move_was_in_remaining_roots',
    ]
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    changed = sum(r['move_changed'] for r in rows)
    print(f'\n=== Results ===')
    print(f'Total turns:       {total}')
    print(f'Move changed:      {changed}/{total} = {100*changed/total:.1f}%')
    print(f'Saved to:          {args.output}')
    print(f'Total time:        {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
