"""
intervention_infer_api.py

Runs a reasoning model with edited reasoning traces as prefill (temperature=0)
and records whether its move changes. Used for Qwen3-235B-A22B-Thinking-2507
and Kimi-K2-Thinking, which were served with vLLM's OpenAI-compatible server.
Same prompt construction as intervention_infer.py: the chat template is opened
with <think>, the edited reasoning is inserted, and </think> is appended.

Usage (vLLM server):
    vllm serve Qwen/Qwen3-235B-A22B-Thinking-2507 --tensor-parallel-size 8 --max-model-len 16384
    python intervention_infer_api.py \
        --model Qwen/Qwen3-235B-A22B-Thinking-2507 \
        --edits results/intervention_qwen3-235b-a22b-thinking/intervention_edits_fd.jsonl \
        --output results/intervention_qwen3-235b-a22b-thinking/intervention_results_fd.csv \
        --api_url http://localhost:8000/v1

Usage (local vLLM):
    python intervention_infer_api.py \
        --model moonshotai/Kimi-K2-Thinking \
        --edits results/intervention_kimi-k2-thinking/intervention_edits_fd.jsonl \
        --output results/intervention_kimi-k2-thinking/intervention_results_fd.csv \
        --tensor_parallel_size 8
"""

import argparse
import json
import re
import csv
import os
import time


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
    """
    Build prompts from edited reasoning traces using the model's chat template.
    """
    all_prompts = []
    all_meta = []
    skipped = 0
    max_input_len = max_model_len - max_new_tokens

    for turn in turns:
        messages = turn['messages']
        edited_reasoning = turn['edited_reasoning']

        if tokenizer is None:
            raise RuntimeError(
                "Tokenizer failed to load; cannot build chat-template prompts. "
                "Install transformers so AutoTokenizer.from_pretrained works."
            )

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

        # If the template did not open the thinking block, open it, then inject the
        # edited reasoning and close with </think> (mirrors intervention_infer.py).
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


def run_local(args, all_prompts, all_meta):
    """Run inference locally using vLLM."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f'Loading model from {args.model} (local vLLM)...')
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
        stop=['<|im_end|>'],
    )

    print(f'Running {len(all_prompts)} inference calls...')
    outputs = llm.generate(all_prompts, sampling_params)

    return outputs


def run_api(args, all_prompts, all_meta):
    """Run inference via OpenAI-compatible API (e.g., a vLLM server).

    Requests are issued concurrently (vLLM batches them server-side); results
    are returned in the original prompt order via ThreadPoolExecutor.map.
    """
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor

    client = OpenAI(api_key=args.api_key, base_url=args.api_url, timeout=180)

    class MockOutput:
        def __init__(self, text):
            self.outputs = [type('obj', (object,), {'text': text})]

    def one(prompt):
        for attempt in range(4):
            try:
                response = client.completions.create(
                    model=args.model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=512,
                    stop=['<|im_end|>'],
                )
                return MockOutput(response.choices[0].text)
            except Exception as e:
                if attempt == 3:
                    print(f'    Error on a prompt (final): {e}')
                    return None
                import time as _t; _t.sleep(2 ** attempt)

    print(f'Connecting to vLLM API at {args.api_url} '
          f'(concurrency={args.concurrency})...')
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        outputs = list(ex.map(one, all_prompts))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='Model name (HF path or local path)')
    parser.add_argument('--edits', required=True,
                        help='JSONL file from intervention_edit.py')
    parser.add_argument('--output', default='results/intervention_results.csv')
    parser.add_argument('--tensor_parallel_size', type=int, default=8,
                        help='TP size for local vLLM (not used with --api_url)')
    parser.add_argument('--max_model_len', type=int, default=16384)
    parser.add_argument('--api_url', default=None,
                        help='OpenAI-compatible API URL (e.g., http://localhost:8000/v1). '
                             'If set, uses API instead of local vLLM.')
    parser.add_argument('--api_key', default='none',
                        help='API key (can be dummy for local vLLM)')
    parser.add_argument('--concurrency', type=int, default=16,
                        help='Concurrent API requests (vLLM batches server-side)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    t_start = time.time()

    turns = load_edits(args.edits)

    if args.api_url:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        except Exception:
            print(f"Warning: could not load tokenizer for {args.model}")
            tokenizer = None
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    all_prompts, all_meta = build_prompts(
        turns, tokenizer, max_model_len=args.max_model_len,
    )

    if args.api_url:
        outputs = run_api(args, all_prompts, all_meta)
    else:
        outputs = run_local(args, all_prompts, all_meta)

    rows = []
    for meta, output in zip(all_meta, outputs):
        if output is None:
            continue

        generated = output.outputs[0].text if output and output.outputs else ""
        new_move = parse_move(generated) or generated.strip()[:20]
        new_coord = move_to_coord(new_move)

        original_coord = meta['chosen_coord']
        pruned_root = meta['pruned_root']
        remaining_roots = [r for r in meta['tree_roots'] if r != pruned_root]

        pruned_suppressed = int(new_coord != pruned_root) if new_coord else int(new_move != meta['original_move'])
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
    print(f'Move changed:      {changed}/{total} = {100*changed/total:.1f}%' if total > 0 else 'N/A')
    print(f'Saved to:          {args.output}')
    print(f'Total time:        {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
