"""
intervention_label.py

For each turn with ≥2 root moves, call Claude to label every paragraph in the
reasoning trace with its structural type, branch_root, and coordinate mentions.
Labels are saved to JSONL for downstream use by intervention_edit.py.

Usage:
    python intervention_label.py \
        --trees_pkl game_trees_df_annotated_preprocessed.pkl \
        --logs_dir aws-logs/four_in_a_row-fen \
        --logs_pattern '*qwen*qwen3*next*80b*' \
        --model_name 'qwen3-next-80b-a3b-thinking' \
        --output results/intervention_labels.jsonl \
        --prune_strategy chosen
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import anthropic


# ── Data loading (shared with intervention_edit.py) ───────────────────────────

def parse_move(text):
    m = re.search(r'<next_move>(.*?)</next_move>', text or '', re.DOTALL)
    return m.group(1).strip() if m else None


def load_jsonl_turns(logs_dir, logs_pattern):
    pattern = os.path.join(logs_dir, '**', logs_pattern)
    files = glob.glob(pattern, recursive=True)
    print(f'Found {len(files)} JSONL files')
    batch_to_turns = {}
    for fpath in files:
        batch_dir = Path(fpath).parent.name
        turns = []
        with open(fpath) as f:
            for line in f:
                d = json.loads(line)
                original_move = parse_move(d['response'].get('content', ''))
                original_reasoning = d['response'].get('reasoning_content', '')
                if not original_move or not original_reasoning:
                    continue
                messages = [
                    {'role': msg['role'], 'content': msg.get('content', '')}
                    for msg in d['input']
                ]
                turns.append({
                    'file': Path(fpath).name,
                    'messages': messages,
                    'original_reasoning': original_reasoning,
                    'original_move': original_move,
                })
        batch_to_turns[batch_dir] = turns
    total = sum(len(v) for v in batch_to_turns.values())
    print(f'Loaded {total} JSONL turns across {len(batch_to_turns)} batches')
    return batch_to_turns


def load_trees(trees_pkl, model_name):
    import pandas as pd
    df = pd.read_pickle(trees_pkl)
    mask = df['model_names'].apply(lambda m: model_name in str(m))
    filtered = df[mask].copy().reset_index(drop=True)
    print(f'Loaded {len(filtered)} rows from game_trees_df '
          f'(model_names contains "{model_name}")')
    return filtered


def extracted_move_to_coord(move_str):
    if not move_str:
        return None
    m = re.match(r'^\s*m?\s*(\d+)\s*[,\s]\s*(\d+)\s*$', str(move_str).strip(), re.IGNORECASE)
    return f"{m.group(1)},{m.group(2)}" if m else None


def select_prune_root(trees, chosen_coord, strategy):
    root_coords = [node[0] for node in trees if isinstance(node, list) and node]
    if strategy == 'chosen':
        return chosen_coord if chosen_coord in root_coords else None
    if strategy == 'non_chosen':
        candidates = [c for c in root_coords if c != chosen_coord]
        if not candidates:
            return None
        sizes = {node[0]: len(node) for node in trees if isinstance(node, list) and node}
        return max(candidates, key=lambda c: sizes.get(c, 0))
    if strategy == 'largest':
        sizes = {node[0]: len(node) for node in trees if isinstance(node, list) and node}
        return max(root_coords, key=lambda c: sizes.get(c, 0)) if root_coords else None
    return None


def tree_to_readable(trees):
    lines = []
    for node in trees:
        if isinstance(node, list) and node:
            root = node[0]
            lines.append(f'  root={root} ({len(node)} nodes)')
    return '\n'.join(lines) if lines else '(no tree data)'


# ── Paragraph splitting ────────────────────────────────────────────────────────

def split_paragraphs(trace):
    paras = []
    last_end = 0
    for sep in re.finditer(r'\n\n+', trace):
        text = trace[last_end:sep.start()]
        if text.strip():
            paras.append((last_end, sep.start(), text))
        last_end = sep.end()
    text = trace[last_end:]
    if text.strip():
        paras.append((last_end, len(trace), text))
    return paras


# ── Claude labeling ────────────────────────────────────────────────────────────

LABEL_SYSTEM = """\
You are assisting a research project studying how chain-of-thought reasoning influences \
LLM behavior. The goal is to surgically remove specific reasoning branches from a model's \
thinking trace and then re-run the model with the edited trace as a prefill — observing \
whether the model changes its decision when it can no longer "see" the reasoning that led \
to a particular move. Accurate labeling is critical: if the wrong paragraphs are removed \
the intervention is invalid; if key paragraphs (especially final confirmations) are missed \
the model will trivially repeat the same answer from the residual context.

You are analyzing a reasoning trace from a Four-in-a-Row game player (4 rows × 9 columns, \
zero-indexed). Coordinates appear as "(r,c)", "row r col c", or "m r c".

The trace has been split into numbered paragraphs. Your task: assign a label to EVERY \
paragraph so that we can later remove the branch for a specific target move.

──────────────────────────────────────────────────
LABEL DEFINITIONS
──────────────────────────────────────────────────
type — one of:
  PREAMBLE         Board parsing, board state description, threat scanning before any \
branch exploration
  BRANCH_START     First paragraph proposing a specific move as the model's candidate \
first move ("If I play X…", "Let me try X…", "What if I place at X…")
  BRANCH_ANALYSIS  Continuation of analysis within a branch (opponent replies, \
counter-replies, evaluations)
  BRANCH_CONCLUSION  Local verdict closing a branch ("X looks strong/weak because…")
  COMPARISON       Cross-branch comparison mentioning multiple candidate moves
  FINAL_DECISION   Confirmation of the chosen move ("I'll play X", "My move is X", \
"Therefore X")
  META             Meta-commentary not tied to a specific move ("Let me think…", \
"First I'll consider…")

branch_root — the depth-0 candidate move this paragraph structurally belongs to, \
as "r,c". Use null for PREAMBLE, COMPARISON, FINAL_DECISION, META. \
Important: assign the same branch_root to ALL paragraphs in a branch, even strategic \
preamble paragraphs that motivate the move before naming it.

mentions — list of coordinates that are part of MOVE SIMULATION only, each with:
  coord  — "r,c"
  depth  — 0=model's move, 1=opponent reply, 2=model counter-reply, etc.

  IMPORTANT: only include a coordinate if it is being proposed or simulated as a future
  move (e.g. "if I play X", "opponent responds at Y", "let me try X"). Do NOT include
  coordinates that merely describe the current board state (e.g. "there are pieces at
  X and Y", "X is occupied", "the board shows pieces at X, Y, Z"). Board-parsing
  references are not move simulations and must be excluded from mentions.

──────────────────────────────────────────────────
OUTPUT FORMAT
──────────────────────────────────────────────────
Return a JSON array with one object per paragraph (in order):
[
  {"para": 0, "type": "PREAMBLE", "branch_root": null, "mentions": []},
  {"para": 1, "type": "BRANCH_START", "branch_root": "1,4", \
"mentions": [{"coord": "1,4", "depth": 0}]},
  ...
]
Return ONLY the JSON array. No explanation, no markdown fences."""


def extract_player_color(messages):
    """Extract the player color ('White' or 'Black') from game messages."""
    for msg in messages:
        content = msg.get('content', '')
        m = re.search(r'You are playing as (White|Black)', content)
        if m:
            return m.group(1)
        m = re.search(r'Current player:\s*(White|Black)', content)
        if m:
            return m.group(1)
    return None


def call_claude_label(client, reasoning_trace, trees, messages=None, max_tokens=16000):
    tree_text = tree_to_readable(trees)
    paras = split_paragraphs(reasoning_trace)
    numbered = '\n\n'.join(f'[{i}] {p}' for i, (_, _, p) in enumerate(paras))

    player_color = extract_player_color(messages) if messages else None
    if player_color:
        opponent_color = 'Black' if player_color == 'White' else 'White'
        color_line = (f"The model is playing as {player_color}. "
                      f"So '{player_color} plays X' = depth 0 (model's move), "
                      f"'{opponent_color} plays Y' = depth 1 (opponent reply).\n\n")
    else:
        color_line = ""

    user_msg = f"""\
{color_line}Search tree (depth-0 = model's candidate first moves):
{tree_text}

Numbered paragraphs of the reasoning trace:
<trace>
{numbered}
</trace>

Label every paragraph with its type, branch_root, and mentions.
Return a JSON array with one object per paragraph."""

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=max_tokens,
        system=LABEL_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        raw = stream.get_final_message().content[0].text.strip()

    try:
        labels = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find('[')
        end = raw.rfind(']')
        if start != -1 and end != -1:
            try:
                labels = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                labels = None
        else:
            labels = None

    return paras, labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trees_pkl', required=True)
    parser.add_argument('--logs_dir', required=True)
    parser.add_argument('--logs_pattern', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--output', default='results/intervention_labels.jsonl')
    parser.add_argument('--prune_strategy', default='chosen',
                        choices=['chosen', 'non_chosen', 'largest'])
    parser.add_argument('--min_roots', type=int, default=2)
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Skip the first N matched turns (0-indexed)')
    parser.add_argument('--max_turns', type=int, default=None)
    parser.add_argument('--max_tokens', type=int, default=32000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    trees_df = load_trees(args.trees_pkl, args.model_name)
    batch_to_turns = load_jsonl_turns(args.logs_dir, args.logs_pattern)

    from collections import defaultdict
    batch_to_rows = defaultdict(list)
    for _, row in trees_df.iterrows():
        batch_dir = Path(row['game_path']).parent.name
        batch_to_rows[batch_dir].append(row)

    matched_pairs = []
    skipped_no_jsonl = 0
    for batch_dir, rows in batch_to_rows.items():
        turns_in_batch = batch_to_turns.get(batch_dir, [])
        rows_sorted = sorted(rows, key=lambda r: r['turn_idx'])
        for pos, row in enumerate(rows_sorted):
            if pos >= len(turns_in_batch):
                skipped_no_jsonl += 1
                continue
            matched_pairs.append((row, turns_in_batch[pos]))

    print(f"Matched {len(matched_pairs)} turns "
          f"({skipped_no_jsonl} skipped: batch not found in JSONL)")

    mismatches = 0
    for row, turn in matched_pairs[:50]:
        dm = re.sub(r'^m\s*', '', str(row['extracted_move'])).strip()
        jm = re.sub(r'^m\s*', '', turn['original_move']).strip()
        if dm != jm:
            mismatches += 1
    print(f"Move-match check on first 50: {50 - mismatches}/50 correct"
          + (" ✓" if mismatches == 0 else f" ✗ {mismatches} mismatches!"))

    matched_pairs = [(row, turn) for row, turn in matched_pairs
                     if row['n_root_moves'] >= args.min_roots]
    print(f"After min_roots≥{args.min_roots} filter: {len(matched_pairs)} turns")

    client = anthropic.Anthropic()
    start = args.start_idx
    end = start + args.max_turns if args.max_turns else len(matched_pairs)
    end = min(end, len(matched_pairs))
    print(f"Processing turns {start}–{end-1} ({end - start} total)")

    saved = 0
    skipped_no_prune = 0
    skipped_api_error = 0

    with open(args.output, 'w') as out_f:
        for i in range(start, end):
            tree_row, jsonl_turn = matched_pairs[i]
            trees = tree_row['trees']
            original_move = jsonl_turn['original_move']
            original_reasoning = jsonl_turn['original_reasoning']
            chosen_coord = extracted_move_to_coord(original_move)
            pruned_root = select_prune_root(trees, chosen_coord, args.prune_strategy)

            if pruned_root is None:
                skipped_no_prune += 1
                continue

            print(f"  [{i+1}/{end}] root={pruned_root} "
                  f"trace={len(original_reasoning)}chars ...", end=' ', flush=True)

            try:
                paras, labels = call_claude_label(
                    client, original_reasoning, trees,
                    messages=jsonl_turn['messages'], max_tokens=args.max_tokens
                )
            except Exception as e:
                print(f"error: {e}")
                skipped_api_error += 1
                continue

            if labels is None:
                print(f"[WARN] could not parse labels")
                skipped_api_error += 1
                continue

            print(f"{len(labels)} paragraphs labeled", flush=True)

            record = {
                'turn_idx_global': i,
                'file': jsonl_turn['file'],
                'messages': jsonl_turn['messages'],
                'original_reasoning': original_reasoning,
                'original_move': original_move,
                'chosen_coord': chosen_coord,
                'pruned_root': pruned_root,
                'prune_strategy': args.prune_strategy,
                'n_root_moves': int(tree_row['n_root_moves']),
                'tree_roots': [node[0] for node in trees if isinstance(node, list) and node],
                'paragraphs': [
                    {'start': s, 'end': e, 'text': t} for s, e, t in paras
                ],
                'labels': labels,
            }
            out_f.write(json.dumps(record) + '\n')
            saved += 1

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{end} turns "
                      f"({saved} saved, {skipped_api_error} errors)")

    print(f"\n=== Done ===")
    print(f"Labeled turns saved: {saved}")
    print(f"Skipped (no prunable root): {skipped_no_prune}")
    print(f"Skipped (API error):        {skipped_api_error}")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
