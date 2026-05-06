"""
run_full.py
===========
Fit on the full dataset (no cross-validation) for one model from
game_trees_df_annotated.pkl.  Use this to obtain final parameter
estimates after cross-validation has confirmed the model generalises.

Usage:
    python run_full.py
    python run_full.py --pkl /path/to/game_trees_df_annotated.pkl --model gpt-5-mini-2025-08-07-high
    python run_full.py --jobid $SLURM_ARRAY_TASK_ID
"""

import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd

from fit import load_and_filter, build_dataset, fit_model, compute_accuracy


# ---------------------------------------------------------------------------
# Myopic: strip tree depth, keep only root actions as leaves
# ---------------------------------------------------------------------------

def make_myopic(dataset: list) -> list:
    """
    Return a copy of the dataset where every tree node's children are stripped.

    Each root action becomes a leaf evaluated at depth 1 (apply move + heuristic),
    with no minimax propagation. The 'board_self' annotations on root nodes are
    preserved so no re-annotation is needed.
    """
    myopic = []
    for sample in dataset:
        myopic_tree = [
            {'move': n['move'], 'children': [], 'board_self': n.get('board_self')}
            for n in sample['tree']
        ]
        myopic.append({**sample, 'tree': myopic_tree})
    return myopic


def make_notree(dataset: list) -> list:
    """
    Return a copy of the dataset where the tree is replaced by ALL legal moves.

    Ignores the LLM's tree entirely: candidate moves are every empty square on
    the board. Each is evaluated as a depth-1 leaf (apply move + heuristic).
    This is the true no-tree baseline — no information from the LLM's reasoning.
    """
    from preprocess import apply_move, EMPTY, ROWS, COLS
    notree = []
    for sample in dataset:
        root_board = sample['root_board']
        notree_tree = [
            {
                'move':       (r, c),
                'children':   [],
                'board_self': apply_move(root_board, r, c),
            }
            for r in range(ROWS) for c in range(COLS)
            if root_board[r, c] == EMPTY
        ]
        notree.append({**sample, 'tree': notree_tree})
    return notree


# ---------------------------------------------------------------------------
# Helpers  (shared with run_cv.py)
# ---------------------------------------------------------------------------

def pick_model(df: pd.DataFrame, requested: str | None) -> str:
    counts = df['model_names'].value_counts()
    print("\n=== Model counts in df ===")
    print(counts.to_string())
    print()

    if requested is not None:
        if requested in counts.index:
            print(f"Using requested model: '{requested}'")
            return requested
        else:
            print(f"WARNING: '{requested}' not found. Falling back to most common.")

    chosen = counts.index[0]
    print(f"Auto-selected model with most rows: '{chosen}' ({counts[chosen]} rows)")
    return chosen


def check_dataset(dataset: list) -> bool:
    if len(dataset) == 0:
        print("\n✗ FAIL: Dataset is empty after filtering. Nothing to fit.")
        return False

    if len(dataset) < 10:
        print(f"\n⚠ WARNING: Only {len(dataset)} samples — fit will be unreliable.")

    n_roots = [len(s['tree']) for s in dataset]
    depths  = []
    for s in dataset:
        def max_depth(node, d=0):
            if not node['children']:
                return d
            return max(max_depth(c, d+1) for c in node['children'])
        depths.append(max(max_depth(n) for n in s['tree']))

    print(f"\n=== Dataset diagnostics ===")
    print(f"  N samples              : {len(dataset)}")
    print(f"  Root actions per turn  : min={min(n_roots)}, "
          f"mean={np.mean(n_roots):.1f}, max={max(n_roots)}")
    print(f"  Tree depth per turn    : min={min(depths)}, "
          f"mean={np.mean(depths):.1f}, max={max(depths)}")

    s0 = dataset[0]
    from minimax import minimax_values
    dummy_w = np.array([1.0, 2.0, 1.5, 5.0, 100.0])
    q = minimax_values(s0['tree'], dummy_w, C=1.5)
    print(f"\n  Spot-check turn 0 (game={s0['game_path'][-40:]}, "
          f"turn={s0['turn_idx']}):")
    print(f"    Root actions + Q-values (dummy weights):")
    for mv, v in sorted(q.items(), key=lambda x: -x[1]):
        marker = " ← chosen" if mv == s0['observed_move'] else ""
        print(f"      {mv}: {v:+.3f}{marker}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl',      default='game_trees_df_annotated.pkl')
    parser.add_argument('--model',    default=None)
    parser.add_argument('--restarts', type=int, default=20)
    parser.add_argument('--col',      default='trees')
    parser.add_argument('--jobid',    default='0')
    parser.add_argument('--myopic',   action='store_true',
                        help='Ignore tree depth: evaluate each root action as a leaf (depth-1 baseline)')
    parser.add_argument('--notree',   action='store_true',
                        help='Ignore tree entirely: evaluate all legal moves as leaves (no-tree baseline)')
    parser.add_argument('--gamma',    action='store_true',
                        help='Fit gamma (minimax discount) as a free parameter')
    args = parser.parse_args()

    print(f"Loading {args.pkl} ...")
    df_all = pd.read_pickle(args.pkl)
    print(f"Total rows: {len(df_all)}")

    args.model = df_all['model_names'].unique()[int(args.jobid)]
    model_name = pick_model(df_all, args.model)

    print(f"\n=== Filtering to model: {model_name} ===")
    df_model = load_and_filter(args.pkl, model_name)
    if len(df_model) == 0:
        print("✗ No rows remain after filtering. Exiting.")
        sys.exit(1)

    print(f"\n=== Building dataset ===")
    dataset = build_dataset(df_model, trees_col=args.col)
    if not check_dataset(dataset):
        sys.exit(1)

    if sum([args.myopic, args.notree, args.gamma]) > 1:
        print("ERROR: --myopic, --notree, and --gamma are mutually exclusive.")
        sys.exit(1)

    if args.myopic:
        print("\n=== Myopic mode: stripping tree depth to depth-1 ===")
        dataset = make_myopic(dataset)
    elif args.notree:
        print("\n=== No-tree mode: replacing tree with all legal moves ===")
        dataset = make_notree(dataset)

    suffix = '_myopic' if args.myopic else ('_notree' if args.notree else ('_gamma' if args.gamma else ''))
    label  = ' [myopic]' if args.myopic else (' [notree]' if args.notree else (' [gamma]' if args.gamma else ''))
    print(f"\n=== Fitting on whole dataset ({args.restarts} restarts){label} ===")

    # gamma model: fit gamma freely; all others: fix gamma=1 (standard minimax)
    fixed_params = {'lam': 0.0}
    if not args.gamma:
        fixed_params['gamma'] = 1.0

    results = fit_model(dataset, n_restarts=args.restarts, seed=42,
                        fixed_params=fixed_params)

    w     = results['weights']
    C     = results['C']
    lam   = results['lam']
    gamma = results['gamma']
    train_acc        = compute_accuracy(dataset, w, C, lam, gamma)
    train_chance_acc = float(np.mean([1.0 / (s['root_board'] == 0).sum() for s in dataset]))
    train_nll_chance = results['nll_chance']

    print(f"\n=== Whole-dataset fit summary ===")
    print(f"  Train NLL/sample : {results['nll'] / len(dataset):.4f}  "
          f"(chance = {train_nll_chance / len(dataset):.4f})")
    print(f"  Train accuracy   : {train_acc:.4f}  "
          f"(chance = {train_chance_acc:.4f})")

    safe_name = model_name.replace('/', '_').replace('@', '_')

    json_path = f"./results/fit_results_full_{safe_name}{suffix}.json"
    summary = {
        'model':                 model_name,
        'n_samples':             len(dataset),
        'weights':               w.tolist(),
        'C':                     C,
        'lam':                   lam,
        'gamma':                 gamma,
        'train_nll_per_sample':  results['nll'] / len(dataset),
        'train_nll_chance':      train_nll_chance,
        'train_accuracy':        train_acc,
        'train_chance_accuracy': train_chance_acc,
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {json_path}")

    pkl_path = f"./results/fit_results_full_{safe_name}{suffix}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Full results saved to {pkl_path}")

    return results


if __name__ == '__main__':
    main()
