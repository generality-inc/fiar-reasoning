"""
Extract search trees from reasoning traces with a GPT-5 judge (paper Section 2.2, Appendix C.2)
and assemble the per-turn dataframe.

    python pipeline/extract_search_trees.py submit   # send every turn's trace to GPT-5 (background responses)
    python pipeline/extract_search_trees.py fetch    # collect finished responses
    python pipeline/extract_search_trees.py build    # parse trees, add tree statistics and game info

submit/fetch need OPENAI_API_KEY. The judge prompt (pipeline/prompts/) is the GEPA-optimized
instruction set reproduced in Appendix C.2. build writes data/game_trees_rebuilt.pkl; the released
data/game_trees.pkl additionally carries 'trees_annotated' (per-node illegal/win/loss/tie tags).
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from fiar.models import API_TO_NAME

LOGS = Path('data/logs/four_in_a_row-fen')
PROMPTS = Path(__file__).parent / 'prompts'
JOBS = Path('data/judge_jobs.jsonl')
RESPONSES = Path('data/judge_responses.jsonl')
JUDGE = dict(model='gpt-5-2025-08-07', temperature=1.0, max_output_tokens=32000,
             reasoning={'effort': 'medium', 'summary': 'detailed'})


def iter_turns(logs=LOGS):
    """Yield (game_path, turn_idx, game, trace) for every turn with a reasoning trace."""
    for path in sorted(logs.rglob('game_log.json')):
        game = json.load(open(path))
        for idx, event in enumerate(game['events']):
            gen = event.get('generation', {})
            if event['type'] == 'turn' and 'reasoning' in gen and 'output' in gen:
                yield str(path.relative_to(logs)), idx, game, gen['reasoning'] + gen['output']


# --- judge -----------------------------------------------------------------

def submit(args):
    from openai import OpenAI
    client = OpenAI(timeout=900)
    system = (PROMPTS / 'extraction_system.txt').read_text().rstrip('\n')
    user = (PROMPTS / 'extraction_user.txt').read_text().rstrip('\n')
    with open(JOBS, 'w') as f:
        for game_path, idx, _, trace in iter_turns():
            r = client.responses.create(instructions=system, input=user.format(trace=trace), background=True,
                                        metadata={'game_path': game_path, 'turn_idx': str(idx)}, **JUDGE)
            f.write(json.dumps({'response_id': r.id, 'game_path': game_path, 'turn_idx': idx}) + '\n')


def fetch(args):
    from openai import OpenAI
    client = OpenAI(timeout=900)
    status = Counter()
    with open(RESPONSES, 'w') as f:
        for line in open(JOBS):
            job = json.loads(line)
            r = client.responses.retrieve(job['response_id'])
            status[r.status] += 1
            if r.status == 'completed':
                f.write(json.dumps({**job, 'text': r.output_text}) + '\n')
    print(dict(status), '- resubmit failed jobs and fetch again if any are not completed')


# --- trees -----------------------------------------------------------------

_COORD = re.compile(r"^\s*'?(-?\d+)\s*,\s*(-?\d+)'?\s*$")
_MOVE = re.compile(r'^\s*m\s*(-?\d+)\s+(-?\d+)\s*$', re.IGNORECASE)


def _coord(label):
    """Normalise 'r,c' or 'm r c' (or a tagged tuple) to 'r,c'; raise ValueError otherwise."""
    if isinstance(label, tuple) and label:
        label = label[0]
    m = isinstance(label, str) and (_COORD.match(label) or _MOVE.match(label))
    if not m:
        raise ValueError(label)
    return f'{int(m.group(1))},{int(m.group(2))}'


def _items(items):
    """Turn a mixed list (subtrees, coordinate strings, bare int pairs) into a list of nodes."""
    out, i = [], 0
    while i < len(items):
        x = items[i]
        if isinstance(x, (list, tuple)) and x:
            out.extend(normalize_forest(x))
        elif isinstance(x, str):
            try:
                out.append([_coord(x)])
            except ValueError:
                pass
        elif isinstance(x, int) and i + 1 < len(items) and isinstance(items[i + 1], int):
            out.append([f'{x},{items[i + 1]}'])
            i += 1
        i += 1
    return out


def normalize_forest(obj):
    """Normalise judge output into a forest of nodes ['r,c', child, ...]."""
    if obj is None or (isinstance(obj, float) and math.isnan(obj)):
        return []
    if isinstance(obj, str):
        try:
            return [[_coord(obj)]]
        except ValueError:
            return []
    if not isinstance(obj, (list, tuple)) or not obj:
        return []
    head = obj[0]
    if isinstance(head, (str, tuple)):
        try:
            return [[_coord(head)] + _items(list(obj[1:]))]
        except ValueError:
            return _items(list(obj))
    if isinstance(head, int) and len(obj) >= 2 and isinstance(obj[1], int):
        return [[f'{head},{obj[1]}'] + _items(list(obj[2:]))]
    return _items(list(obj))


def tree_stats(trees):
    """
    Depth (max ply), breadth (number of depth-1 nodes) and size of the judge's raw forest
    (before normalize_forest; depth-1 nodes are at ply 1). Nodes are ['r,c', child, ...] or [r, c, child, ...].
    """
    def children(node):
        if not isinstance(node, (list, tuple)) or not node:
            return []
        rest = node[1:] if isinstance(node[0], str) else node[2:]
        return [ch for ch in rest if isinstance(ch, (list, tuple))]

    if not isinstance(trees, (list, tuple)):
        trees = []
    levels, stack = Counter(), [(t, 1) for t in trees if isinstance(t, (list, tuple))]
    while stack:
        node, depth = stack.pop()
        levels[depth] += 1
        stack.extend((child, depth + 1) for child in children(node))
    return {'max_ply': max(levels, default=0), 'n_root_moves': len(trees), 'total_nodes': sum(levels.values())}


# --- game info -------------------------------------------------------------

def model_id(player):
    """API identifier with reasoning-effort suffix, as used in API_TO_NAME."""
    name, config = player.get('model', ''), player.get('config') or {}
    if name == 'deepseek-ai/deepseek-v3.1-maas' and config.get('thinking'):
        return 'accounts/fireworks/models/deepseek-v3p1-medium'
    if name == 'deepseek-ai/deepseek-r1-0528-maas':
        return 'accounts/fireworks/models/deepseek-r1-0528-medium'
    name = {'claude-opus-4-1@20250805': 'claude-opus-4-1-20250805',
            'claude-sonnet-4@20250514': 'claude-sonnet-4-20250514'}.get(name, name)
    effort = config.get('reasoning_effort')
    return f'{name}-{effort}' if effort and not name.endswith(f'-{effort}') else name


def game_outcome(game):
    """1 if player 0 won, 0 if player 1 won, 0.5 for a draw (forfeits count as losses)."""
    result = game['result']
    if result['type'] == 'draw':
        return 0.5
    return {0: 1.0, 1: 0.0}.get(result['metadata'].get('winner'), 0.5)


def board_state(game, idx):
    """The board shown to the player at turn idx, rebuilt by replaying the legal moves before it."""
    board = [['.'] * 9 for _ in range(4)]
    events = game['events']
    for i, e in enumerate(events[:idx]):
        if e['type'] == 'turn' and events[i + 1]['type'] == 'move_success':
            r, c = map(int, e['extracted_move'].split()[1:])
            board[r][c] = game['players'][int(e['player'])]['name'][0]
    fen = '/'.join(re.sub(r'\.+', lambda m: str(len(m.group())), ''.join(row)) for row in board)
    if fen == '9/9/9/9':
        return None                     # opening turns carry no board_state in the released data
    colour = game['players'][int(events[idx]['player'])]['name']
    return f'FEN: {fen}\n\nCurrent player: {colour} ({colour[0]})\n'


def build(args):
    from json_repair import repair_json
    rows = []
    for line in open(RESPONSES):
        r = json.loads(line)
        game = json.load(open(LOGS / r['game_path']))
        events, idx = game['events'], r['turn_idx']
        player = int(events[idx]['player'])
        name = API_TO_NAME[model_id(game['players'][player])]
        raw = json.loads(repair_json(r['text']))['trees']
        outcome = game_outcome(game)
        rows.append({
            'game_path': r['game_path'], 'turn_idx': idx, 'model_names': name, 'trees': normalize_forest(raw),
            'extracted_move': events[idx].get('extracted_move'),
            'move_success': events[idx + 1]['type'] == 'move_success',
            **tree_stats(raw),
            'board_state': board_state(game, idx),
            'outcome_perspective': outcome if player == 0 else 1 - outcome,
        })
    df = pd.DataFrame(rows)
    df.to_pickle('data/game_trees_rebuilt.pkl')
    print(f'{len(df)} turns -> data/game_trees_rebuilt.pkl')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['submit', 'fetch', 'build'])
    args = ap.parse_args()
    {'submit': submit, 'fetch': fetch, 'build': build}[args.step](args)
