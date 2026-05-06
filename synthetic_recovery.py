"""
synthetic_recovery.py
=====================
Model recovery test: simulate move data from a full-tree-backup agent (gamma=1)
using real LLM game trees, then fit the model and verify gamma is recovered.

Addresses the question: "IF the true decision mechanism were full-tree backup,
would our fitting procedure correctly identify it?"

Procedure
---------
1. Load real LLM game trees (same board positions and tree structures as in the
   actual analysis, so the test is ecologically valid).
2. Sample synthetic moves from the full-tree-backup model (gamma=1) using
   specified true parameters.
3. Fit three models on the synthetic data:
   - Full model   : gamma free (should recover gamma ≈ 1)
   - Myopic model : gamma fixed to 0 (depth-1 baseline; should be worse)
   - Oracle model : gamma fixed to 1 (knows the true value; upper-bound fit)
4. Compare NLL and recovered parameters.

Usage
-----
    python synthetic_recovery.py \\
        --pkl game_trees_df_annotated.pkl \\
        --model accounts/fireworks/models/qwen3-235b-a22b-thinking-2507-medium \\
        --n_samples 500 \\
        --n_restarts 10 \\
        --seed 42

True parameters default to the fitted values of the specified model with
gamma forced to 1 for the simulation.
"""

import argparse
import json
import sys

import numpy as np
import scipy.special

from fit import load_and_filter, build_dataset, fit_model, neg_log_likelihood
from minimax import minimax_values
from run_full import make_myopic


# ---------------------------------------------------------------------------
# Sampling from the true model
# ---------------------------------------------------------------------------

def sample_synthetic_moves(dataset: list,
                            weights: np.ndarray,
                            C: float,
                            gamma: float,
                            lam: float,
                            rng: np.random.Generator) -> list:
    """
    Replace observed_move in every sample with one drawn from the true model.

    The true model is the mixture:
        P(a) = (1 - lam) * softmax(Q(a)) + lam / |A|
    where Q-values are full-tree-backup minimax values with the given gamma.

    The sampled move is always a root action in the tree by construction.
    """
    syn = []
    for sample in dataset:
        action_values = minimax_values(sample['tree'], weights, C, gamma)
        moves  = list(action_values.keys())
        qvals  = np.array([action_values[mv] for mv in moves])
        n      = len(moves)
        probs  = (1.0 - lam) * scipy.special.softmax(qvals) + lam / n
        idx    = rng.choice(n, p=probs)
        syn.append({**sample, 'observed_move': moves[idx]})
    return syn


# ---------------------------------------------------------------------------
# NLL helpers
# ---------------------------------------------------------------------------

def _params_to_vec(weights, C, lam, gamma):
    return np.concatenate([
        weights,
        [np.log(C),
         scipy.special.logit(np.clip(lam,   1e-6, 1 - 1e-6)),
         scipy.special.logit(np.clip(gamma, 1e-6, 1 - 1e-6))],
    ])


def eval_nll(dataset, weights, C, lam, gamma):
    return neg_log_likelihood(_params_to_vec(weights, C, lam, gamma), dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl',         default='game_trees_df_annotated.pkl')
    parser.add_argument('--model',       default=None,
                        help='Model name in the dataframe')
    parser.add_argument('--n_samples',   type=int, default=None,
                        help='Cap on number of dataset samples (default: all)')
    parser.add_argument('--n_restarts',  type=int, default=10)
    parser.add_argument('--seed',        type=int, default=42)
    # True parameter overrides (defaults: values typical of a thinking model)
    parser.add_argument('--true_weights', type=float, nargs=5,
                        default=[0.42, 0.26, 0.18, 0.71, 4.80],
                        metavar=('WC', 'WC2', 'WUC2', 'W3', 'W4'),
                        help='True heuristic weights [centre c2 uc2 three four]')
    parser.add_argument('--true_C',     type=float, default=3.08)
    parser.add_argument('--true_lam',   type=float, default=0.05)
    parser.add_argument('--true_gamma', type=float, default=1.0,
                        help='True gamma (1.0 = full-tree backup)')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # 1. Load and build dataset
    # ------------------------------------------------------------------
    import pandas as pd
    print(f"Loading {args.pkl} ...")
    df_all = pd.read_pickle(args.pkl)
    print(f"Total rows: {len(df_all)}")

    if args.model is None:
        args.model = df_all['model_names'].value_counts().index[0]
        print(f"Auto-selected model: '{args.model}'")

    df_model = load_and_filter(args.pkl, args.model)
    if len(df_model) == 0:
        print("No rows after filtering. Exiting.")
        sys.exit(1)

    dataset = build_dataset(df_model)
    if len(dataset) == 0:
        print("Empty dataset. Exiting.")
        sys.exit(1)

    if args.n_samples is not None and args.n_samples < len(dataset):
        idx = rng.choice(len(dataset), size=args.n_samples, replace=False)
        dataset = [dataset[i] for i in sorted(idx)]
        print(f"Subsampled to {len(dataset)} turns")

    # ------------------------------------------------------------------
    # 2. Sample synthetic moves under true full-tree-backup model
    # ------------------------------------------------------------------
    w_true    = np.array(args.true_weights)
    C_true    = args.true_C
    lam_true  = args.true_lam
    gamma_true = args.true_gamma

    print(f"\n=== True parameters ===")
    print(f"  weights  = {w_true}")
    print(f"  C        = {C_true:.3f}")
    print(f"  lam      = {lam_true:.3f}")
    print(f"  gamma    = {gamma_true:.3f}  ← full-tree backup")

    syn_dataset = sample_synthetic_moves(
        dataset, w_true, C_true, gamma_true, lam_true, rng
    )

    # Sanity: every sampled move must be a root action
    for s in syn_dataset:
        assert s['observed_move'] in [n['move'] for n in s['tree']]

    # Oracle NLL (true params evaluated on synthetic data)
    oracle_nll = eval_nll(syn_dataset, w_true, C_true, lam_true, gamma_true)
    nll_chance = sum(np.log(len(s['tree'])) for s in syn_dataset)
    print(f"\n  Oracle NLL/sample = {oracle_nll / len(syn_dataset):.4f}  "
          f"(chance = {nll_chance / len(syn_dataset):.4f})")

    # ------------------------------------------------------------------
    # 3. Fit three models on the synthetic data
    # ------------------------------------------------------------------

    print(f"\n{'='*60}")
    print(f"=== Fit 1: Full model (gamma free) ===")
    print(f"{'='*60}")
    results_full = fit_model(
        syn_dataset,
        n_restarts=args.n_restarts,
        seed=args.seed,
        fixed_params={'lam': 0.0},   # fix lam=0 to match run_full.py convention
    )

    print(f"\n{'='*60}")
    print(f"=== Fit 2: Myopic model (gamma = 0, depth-1 only) ===")
    print(f"{'='*60}")
    myopic_dataset = make_myopic(syn_dataset)
    results_myopic = fit_model(
        myopic_dataset,
        n_restarts=args.n_restarts,
        seed=args.seed,
        fixed_params={'lam': 0.0, 'gamma': 0.0},
    )

    print(f"\n{'='*60}")
    print(f"=== Fit 3: Oracle model (gamma = 1, true value fixed) ===")
    print(f"{'='*60}")
    results_oracle = fit_model(
        syn_dataset,
        n_restarts=args.n_restarts,
        seed=args.seed,
        fixed_params={'lam': 0.0, 'gamma': 1.0},
    )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    N = len(syn_dataset)

    def _nll_per(r): return r['nll'] / N

    print(f"\n{'='*60}")
    print(f"=== Recovery Summary (N={N} synthetic turns) ===")
    print(f"{'='*60}")
    print(f"\nTrue parameters:")
    print(f"  gamma = {gamma_true:.3f}  (full-tree backup)")
    print(f"  w     = {w_true}")
    print(f"  C     = {C_true:.3f}  lam = {lam_true:.3f}")

    print(f"\nFitted gamma:")
    print(f"  Full model  : gamma = {results_full['gamma']:.4f}  "
          f"(target = {gamma_true:.3f})")
    print(f"  Myopic      : gamma = 0.0000  (fixed)")
    print(f"  Oracle      : gamma = 1.0000  (fixed)")

    print(f"\nFitted weights comparison (true vs full-model recovery):")
    wnames = ['w_centre', 'w_c2', 'w_uc2', 'w_three', 'w_four']
    for name, wt, wr in zip(wnames, w_true, results_full['weights']):
        print(f"  {name:<12s}  true={wt:+.3f}  recovered={wr:+.3f}")
    print(f"  {'C':<12s}  true={C_true:+.3f}  recovered={results_full['C']:+.3f}")

    print(f"\nNLL per sample (lower is better):")
    print(f"  {'Model':<18s}  {'NLL/sample':>12s}  {'ΔNLL vs chance':>16s}")
    print(f"  {'-'*50}")
    models = [
        ('Full (gamma free)', results_full),
        ('Myopic (gamma=0)', results_myopic),
        ('Oracle (gamma=1)', results_oracle),
    ]
    for name, res in models:
        nll_s    = _nll_per(res)
        delta    = nll_chance / N - nll_s
        print(f"  {name:<18s}  {nll_s:>12.4f}  {delta:>+16.4f}")
    print(f"  {'Chance':<18s}  {nll_chance/N:>12.4f}  {'0.0000':>16s}")

    delta_myopic = _nll_per(results_myopic) - _nll_per(results_full)
    print(f"\nΔNLL (myopic − full): {delta_myopic:+.4f} per sample  "
          f"({'full model wins' if delta_myopic > 0 else 'UNEXPECTED: myopic wins'})")

    print(f"\nConclusion:")
    if abs(results_full['gamma'] - gamma_true) < 0.2 and delta_myopic > 0.01:
        print(f"  ✓ Recovery successful: gamma≈{results_full['gamma']:.2f} "
              f"(true={gamma_true:.1f}), full model better than myopic by "
              f"{delta_myopic:.4f} NLL/sample")
    else:
        print(f"  ✗ Recovery ambiguous: gamma={results_full['gamma']:.3f} "
              f"(true={gamma_true:.1f}), ΔNLL={delta_myopic:.4f}")

    # Save results
    summary = {
        'n_samples':      N,
        'true_gamma':     gamma_true,
        'true_C':         C_true,
        'true_lam':       lam_true,
        'true_weights':   w_true.tolist(),
        'oracle_nll_per_sample':      oracle_nll / N,
        'chance_nll_per_sample':      nll_chance / N,
        'full_model': {
            'gamma':          results_full['gamma'],
            'C':              results_full['C'],
            'weights':        results_full['weights'].tolist(),
            'nll_per_sample': _nll_per(results_full),
        },
        'myopic_model': {
            'gamma':          0.0,
            'C':              results_myopic['C'],
            'weights':        results_myopic['weights'].tolist(),
            'nll_per_sample': _nll_per(results_myopic),
        },
        'oracle_model': {
            'gamma':          1.0,
            'C':              results_oracle['C'],
            'weights':        results_oracle['weights'].tolist(),
            'nll_per_sample': _nll_per(results_oracle),
        },
    }
    out_path = 'results/synthetic_recovery.json'
    import os
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
