"""
intervention/infer_vllm.py

Same as intervention/infer_local.py, but against a model served by vLLM's
OpenAI-compatible server. Used for Qwen3-235B-A22B-Thinking-2507 and
Kimi-K2-Thinking. Prompts are built locally with the model's chat template and
sent to the raw completions endpoint.

Usage:
    vllm serve moonshotai/Kimi-K2-Thinking --tensor-parallel-size 8 \
        --max-model-len 16384 --trust-remote-code
    python intervention/infer_vllm.py \
        --model moonshotai/Kimi-K2-Thinking \
        --edits results/intervention/kimi-k2-thinking/edits_fd.jsonl \
        --output results/intervention/kimi-k2-thinking/results_fd.csv
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

from infer_local import build_prompts, load_edits, write_results


def run_api(args, all_prompts):
    """Issue completions concurrently (vLLM batches server-side); keeps prompt order."""
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key, base_url=args.api_url, timeout=180)

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
                return response.choices[0].text
            except Exception as e:
                if attempt == 3:
                    print(f'    Error on a prompt (final): {e}')
                    return None
                time.sleep(2 ** attempt)

    print(f'Connecting to vLLM API at {args.api_url} '
          f'(concurrency={args.concurrency})...')
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        return list(ex.map(one, all_prompts))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', required=True,
                        help='Served model name (HF id); also used to load the tokenizer')
    parser.add_argument('--edits', required=True,
                        help='edits_<strategy>.jsonl from intervention/edit.py')
    parser.add_argument('--output', required=True,
                        help='results_<strategy>.csv to write')
    parser.add_argument('--max_model_len', type=int, default=16384)
    parser.add_argument('--api_url', default='http://localhost:8000/v1',
                        help='OpenAI-compatible endpoint of the vLLM server')
    parser.add_argument('--api_key', default='none',
                        help='API key (dummy for a local vLLM server)')
    parser.add_argument('--concurrency', type=int, default=16,
                        help='Concurrent requests')
    args = parser.parse_args()

    from transformers import AutoTokenizer

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    t_start = time.time()

    turns = load_edits(args.edits)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    all_prompts, all_meta = build_prompts(
        turns, tokenizer, max_model_len=args.max_model_len,
    )

    generations = run_api(args, all_prompts)
    write_results(all_meta, generations, args.output)
    print(f'Total time:        {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
