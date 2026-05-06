"""
run_cv.py
=========
Run k-fold cross-validation fitting on one model from game_trees_df_annotated.pkl.

Usage:
    python run_cv.py
    python run_cv.py --pkl /path/to/game_trees_df_annotated.pkl --model gpt-5-mini-2025-08-07-high
    python run_cv.py --jobid $SLURM_ARRAY_TASK_ID
"""

import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd

from fit import load_and_filter, build_dataset, cross_validate


# ---------------------------------------------------------------------------
# Helpers
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
    parser.add_argument('--folds',    type=int, default=10)
    parser.add_argument('--col',      default='trees')
    parser.add_argument('--jobid',    default='0')
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

    print(f"\n=== Cross-validation ({args.folds} folds, {args.restarts} restarts per fold) ===")
    cv = cross_validate(dataset, n_folds=args.folds, n_restarts=args.restarts, seed=42)

    safe_name = model_name.replace('/', '_').replace('@', '_')

    json_path = f"./results/fit_results_{safe_name}.json"
    summary = {
        'model':     model_name,
        'n_samples': len(dataset),
        'cv': {
            'mean_test_nll':        cv['mean_test_nll'],
            'mean_test_nll_chance': cv['mean_test_nll_chance'],
            'mean_test_accuracy':   cv['mean_test_accuracy'],
            'mean_test_chance_acc': cv['mean_test_chance_acc'],
            'folds': [
                {k: v.tolist() if hasattr(v, 'tolist') else v
                 for k, v in f.items() if k != 'weights'}
                | {'weights': f['weights'].tolist()}
                for f in cv['fold_results']
            ],
        },
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nCV results saved to {json_path}")

    pkl_path = f"./results/fit_results_{safe_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(cv, f)
    print(f"Full CV results saved to {pkl_path}")

    return cv


if __name__ == '__main__':
    main()
