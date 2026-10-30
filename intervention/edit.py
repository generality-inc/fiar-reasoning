"""
intervention/edit.py

Generates the seven edited reasoning traces used in the causal intervention
(Figure 4). Each strategy removes the FINAL_DECISION paragraphs plus:

  fd                             — nothing else
  fd_branch_comp                 — the whole chosen branch (and comparisons naming it)
  bc_minus_d0only                — same, but keep paragraphs with only depth-0 mentions
  bc_minus_d0_and_deep1plus      — same, but keep paragraphs with only depth-0 or only depth≥1 mentions

The *_ctrl variants target the most-labeled unchosen branch instead.

Usage:
    python intervention/edit.py --model kimi-k2-thinking
"""

import argparse
import json
import os
import re
from collections import Counter


STRATEGIES = [
    'fd',
    'fd_branch_comp',
    'fd_branch_comp_ctrl',
    'bc_minus_d0only',
    'bc_minus_d0only_ctrl',
    'bc_minus_d0_and_deep1plus',
    'bc_minus_d0_and_deep1plus_ctrl',
]

# Which depth classes are KEPT for bc_minus_* strategies
KEEP_CLASSES = {
    'bc_minus_d0only':               {'d0only'},
    'bc_minus_d0only_ctrl':          {'d0only'},
    'bc_minus_d0_and_deep1plus':     {'d0only', 'deep1plus'},
    'bc_minus_d0_and_deep1plus_ctrl':{'d0only', 'deep1plus'},
}


def _coord_in_text(text, root):
    pr, pc = root.split(',')
    patterns = [
        rf'\bm\s+{pr}\s+{pc}\b',
        rf'\({pr}\s*,\s*{pc}\)',
        rf'\b{pr}\s*,\s*{pc}\b',
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def pick_control_root(record):
    pruned_root = record['pruned_root']
    counts = Counter()
    for lbl in record['labels']:
        if not isinstance(lbl, dict):
            continue
        br = lbl.get('branch_root')
        if br and br != pruned_root:
            counts[br] += 1
    return counts.most_common(1)[0][0] if counts else None


def _mention_depth_class(label):
    mentions = [m for m in label.get('mentions', []) if isinstance(m, dict)]
    if not mentions:
        return None
    has_d0   = any(m.get('depth', 0) == 0 for m in mentions)
    has_deep = any(m.get('depth', 0) >= 1 for m in mentions)
    if has_d0 and not has_deep:
        return 'd0only'
    if has_deep and not has_d0:
        return 'deep1plus'
    return 'd0_and_deep'


def _is_related_to_root(lbl, paras, idx, target_root):
    ltype = lbl.get('type', '')
    if ltype == 'COMPARISON':
        mentions = [m.get('coord') for m in lbl.get('mentions', [])
                    if isinstance(m, dict)]
        if target_root in mentions:
            return True
        if idx < len(paras):
            return _coord_in_text(paras[idx]['text'], target_root)
        return False
    return lbl.get('branch_root') == target_root


def _apply_edits(original, paras, to_remove, max_frac):
    if not to_remove:
        return original, [], 0

    removed_chars = sum(
        paras[i]['end'] - paras[i]['start']
        for i in to_remove if i < len(paras)
    )
    if removed_chars / max(len(original), 1) > max_frac:
        print(f"    [WARN] would remove {removed_chars/len(original):.0%} — skipping")
        return original, [], 0

    spans = sorted((paras[i]['start'], paras[i]['end'])
                   for i in to_remove if i < len(paras))
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])

    removed_segments = [original[s:e] for s, e in merged]
    edited = original
    for s, e in reversed(merged):
        edited = edited[:s] + edited[e:]
    edited = re.sub(r'\n{3,}', '\n\n', edited).strip()
    return edited, removed_segments, len(to_remove)


def apply_strategy(record, strategy, control_root=None, max_frac=0.85):
    if '_ctrl' in strategy and control_root is None:
        return None

    original    = record['original_reasoning']
    paras       = record['paragraphs']
    labels      = record['labels']
    pruned_root = record['pruned_root']
    target_root = control_root if '_ctrl' in strategy else pruned_root

    label_map = {lbl['para']: lbl for lbl in labels if isinstance(lbl, dict)}
    to_remove = set()

    if strategy == 'fd':
        # Remove only FINAL_DECISION
        for idx, lbl in label_map.items():
            if idx < len(paras) and lbl.get('type') == 'FINAL_DECISION':
                to_remove.add(idx)

    elif strategy in ('fd_branch_comp', 'fd_branch_comp_ctrl'):
        # Remove FINAL_DECISION + all branch/comp paragraphs for target_root
        for idx, lbl in label_map.items():
            if idx >= len(paras):
                continue
            ltype = lbl.get('type')
            if ltype == 'FINAL_DECISION':
                to_remove.add(idx)
            elif _is_related_to_root(lbl, paras, idx, target_root):
                to_remove.add(idx)

    else:
        # bc_minus_* : remove FINAL_DECISION + branch/comp for target_root
        # EXCEPT paragraphs whose depth class is in the keep set
        keep = KEEP_CLASSES[strategy]
        for idx, lbl in label_map.items():
            if idx >= len(paras):
                continue
            ltype = lbl.get('type')
            if ltype == 'FINAL_DECISION':
                to_remove.add(idx)
                continue
            if not _is_related_to_root(lbl, paras, idx, target_root):
                continue
            if _mention_depth_class(lbl) not in keep:
                to_remove.add(idx)

    return _apply_edits(original, paras, to_remove, max_frac)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model',      default=None,
                        help='Shorthand: read/write results/intervention/<model>/')
    parser.add_argument('--labels',     default=None,
                        help='Labels JSONL (overrides --model)')
    parser.add_argument('--output_dir', default=None,
                        help='Output folder (overrides --model)')
    parser.add_argument('--max_frac',   type=float, default=0.85,
                        help='Skip edits that would remove more than this fraction of the trace')
    args = parser.parse_args()
    if args.model:
        model_dir = os.path.join('results/intervention', args.model)
        args.labels = args.labels or os.path.join(model_dir, 'labels.jsonl')
        args.output_dir = args.output_dir or model_dir
    if not (args.labels and args.output_dir):
        parser.error('pass --model, or both --labels and --output_dir')

    os.makedirs(args.output_dir, exist_ok=True)

    records = [json.loads(l) for l in open(args.labels)]
    print(f"Loaded {len(records)} labeled turns from {args.labels}")

    out_files = {
        s: open(os.path.join(args.output_dir, f'edits_{s}.jsonl'), 'w')
        for s in STRATEGIES
    }
    summary = {s: {'n': 0, 'skipped': 0, 'no_ctrl': 0,
                   'total_chars': 0, 'total_paras': 0}
               for s in STRATEGIES}

    for record in records:
        control_root = pick_control_root(record)

        for strategy in STRATEGIES:
            result = apply_strategy(record, strategy,
                                    control_root=control_root,
                                    max_frac=args.max_frac)
            if result is None:
                summary[strategy]['no_ctrl'] += 1
                continue

            edited, removed_segs, n_paras = result
            chars_removed = len(record['original_reasoning']) - len(edited)

            if chars_removed == 0:
                summary[strategy]['skipped'] += 1
            summary[strategy]['n']           += 1
            summary[strategy]['total_chars'] += chars_removed
            summary[strategy]['total_paras'] += n_paras

            out = {
                'turn_idx_global':    record['turn_idx_global'],
                'file':               record['file'],
                'strategy':           strategy,
                'messages':           record['messages'],
                'original_reasoning': record['original_reasoning'],
                'edited_reasoning':   edited,
                'removed_segments':   removed_segs,
                'n_segments_removed': len(removed_segs),
                'n_paras_removed':    n_paras,
                'chars_removed':      chars_removed,
                'original_move':      record['original_move'],
                'chosen_coord':       record['chosen_coord'],
                'pruned_root':        record['pruned_root'],
                'control_root':       control_root,
                'prune_strategy':     record['prune_strategy'],
                'n_root_moves':       record['n_root_moves'],
                'tree_roots':         record['tree_roots'],
            }
            out_files[strategy].write(json.dumps(out) + '\n')

    for f in out_files.values():
        f.close()

    print(f"\n{'Strategy':<35s} {'n':>5} {'skipped':>8} {'no_ctrl':>8} "
          f"{'avg_chars':>10} {'avg_paras':>10}")
    print('-' * 78)
    for s in STRATEGIES:
        st = summary[s]
        n = st['n']
        print(f"{s:<35s} {n:>5} {st['skipped']:>8} {st['no_ctrl']:>8} "
              f"{st['total_chars']/max(n,1):>10.0f} "
              f"{st['total_paras']/max(n,1):>10.1f}")
    print(f"\nOutput written to {args.output_dir}/")


if __name__ == '__main__':
    main()
