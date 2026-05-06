"""
run_model_recovery.py
=====================
Systematic model recovery test across all LLMs.

For each model with fitted results:
  1. Simulate data from the full-tree model (gamma=1, using fitted full-tree params)
     → Fit full-tree and myopic; full-tree should win
  2. Simulate data from the myopic model (gamma=0, using fitted myopic params)
     → Fit full-tree and myopic; myopic should win

Results are saved per-model to results/recovery/ and aggregated to
results/model_recovery_all.json.

Usage (single model):
    python run_model_recovery.py --model deepseek-reasoner

Usage (SLURM array, one job per model):
    python run_model_recovery.py --jobid $SLURM_ARRAY_TASK_ID

Usage (all models sequentially):
    python run_model_recovery.py --all
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import scipy.special

from fit import load_and_filter, build_dataset, fit_model, neg_log_likelihood
from minimax import minimax_values
from run_full import make_myopic


# ── Helpers ────────────────────────────────────────────────────────────────

RESULTS_DIR = 'results/fit_results'
RECOVERY_DIR = 'results/recovery'
PKL = 'game_trees_df_annotated.pkl'

OPEN_SOURCE_KEYWORDS = ['deepseek', 'glm', 'kimi', 'llama', 'qwen', 'oss']

def is_open_source(sname):
    n = sname.lower()
    return any(k in n for k in OPEN_SOURCE_KEYWORDS)


def safe_name(model):
    return model.replace('/', '_').replace('@', '_')


def get_all_models(open_source_only=False):
    """Return list of safe model names with both full and myopic fit files."""
    pattern = os.path.join(RESULTS_DIR, 'fit_results_full_*.json')
    files = glob.glob(pattern)
    models = []
    for f in sorted(files):
        base = os.path.basename(f)
        if any(base.endswith(s) for s in ['_gamma.json', '_myopic.json', '_notree.json']):
            continue
        sname = base.replace('fit_results_full_', '').replace('.json', '')
        myopic_f = f.replace('.json', '_myopic.json')
        if not os.path.exists(myopic_f):
            continue
        if open_source_only and not is_open_source(sname):
            continue
        models.append(sname)
    return models


def load_fit(sname, suffix=''):
    path = os.path.join(RESULTS_DIR, f'fit_results_full_{sname}{suffix}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def sample_synthetic_moves(dataset, weights, C, gamma, lam, rng):
    """Sample one move per trial from the softmax model."""
    syn = []
    for sample in dataset:
        action_values = minimax_values(sample['tree'], weights, C, gamma)
        moves = list(action_values.keys())
        qvals = np.array([action_values[mv] for mv in moves])
        n = len(moves)
        probs = (1.0 - lam) * scipy.special.softmax(qvals) + lam / n
        idx = rng.choice(n, p=probs)
        syn.append({**sample, 'observed_move': moves[idx]})
    return syn


def eval_nll_per_sample(dataset, weights, C, lam, gamma):
    params = np.concatenate([
        weights,
        [np.log(C),
         scipy.special.logit(np.clip(lam,   1e-6, 1 - 1e-6)),
         scipy.special.logit(np.clip(gamma, 1e-6, 1 - 1e-6))],
    ])
    return neg_log_likelihood(params, dataset) / len(dataset)


def run_recovery_for_model(sname, n_restarts=5, seed=42, n_samples=None):
    """
    Run model recovery for one model.
    Returns a dict with NLL per sample for all conditions, or None on failure.
    """
    full_fit   = load_fit(sname, '')
    myopic_fit = load_fit(sname, '_myopic')
    if full_fit is None or myopic_fit is None:
        print(f'  SKIP {sname}: missing fit files')
        return None

    model_api = full_fit['model']
    rng = np.random.default_rng(seed)

    # Load dataset
    dataset = build_dataset(load_and_filter(PKL, model_api))
    if len(dataset) < 20:
        print(f'  SKIP {sname}: only {len(dataset)} samples')
        return None

    if n_samples and n_samples < len(dataset):
        idx = rng.choice(len(dataset), size=n_samples, replace=False)
        dataset = [dataset[i] for i in sorted(idx)]

    N = len(dataset)
    print(f'  N={N} samples')

    w_full   = np.array(full_fit['weights']);   C_full   = full_fit['C']
    w_myopic = np.array(myopic_fit['weights']); C_myopic = myopic_fit['C']
    lam = 0.0  # consistent with how models were originally fit

    results = {'model': model_api, 'sname': sname, 'N': N}

    # ── Condition 1: Simulate from full-tree (gamma=1, original trees) ────
    print(f'  Simulating from full-tree model...')
    syn_full = sample_synthetic_moves(dataset, w_full, C_full, 1.0, lam, rng)

    print(f'    Fitting full-tree model...')
    fit_ff = fit_model(syn_full, n_restarts=n_restarts, seed=seed,
                       fixed_params={'lam': 0.0, 'gamma': 1.0})
    print(f'    Fitting myopic model...')
    fit_fm = fit_model(make_myopic(syn_full), n_restarts=n_restarts, seed=seed,
                       fixed_params={'lam': 0.0, 'gamma': 1.0})

    results['sim_full'] = {
        'nll_full':   fit_ff['nll'] / N,
        'nll_myopic': fit_fm['nll'] / N,
        'delta':      (fit_fm['nll'] - fit_ff['nll']) / N,  # positive = full wins
    }
    print(f'    full={fit_ff["nll"]/N:.4f}  myopic={fit_fm["nll"]/N:.4f}  '
          f'delta={results["sim_full"]["delta"]:+.4f}')

    # ── Condition 2: Simulate from myopic (stripped trees, gamma=1) ─────
    print(f'  Simulating from myopic model...')
    syn_myopic = sample_synthetic_moves(
        make_myopic(dataset), w_myopic, C_myopic, 1.0, lam, rng
    )
    # Attach original (full) trees back for full-tree fitting
    syn_myopic_full_trees = [
        {**orig, 'observed_move': syn['observed_move']}
        for orig, syn in zip(dataset, syn_myopic)
    ]

    print(f'    Fitting full-tree model...')
    fit_mf = fit_model(syn_myopic_full_trees, n_restarts=n_restarts, seed=seed,
                       fixed_params={'lam': 0.0, 'gamma': 1.0})
    print(f'    Fitting myopic model...')
    fit_mm = fit_model(make_myopic(syn_myopic_full_trees), n_restarts=n_restarts,
                       seed=seed, fixed_params={'lam': 0.0, 'gamma': 1.0})

    results['sim_myopic'] = {
        'nll_full':   fit_mf['nll'] / N,
        'nll_myopic': fit_mm['nll'] / N,
        'delta':      (fit_mm['nll'] - fit_mf['nll']) / N,  # negative = myopic wins
    }
    print(f'    full={fit_mf["nll"]/N:.4f}  myopic={fit_mm["nll"]/N:.4f}  '
          f'delta={results["sim_myopic"]["delta"]:+.4f}')

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    global PKL
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       default=None,
                        help='Safe model name (e.g. deepseek-reasoner)')
    parser.add_argument('--jobid',       type=int, default=None,
                        help='SLURM array job index (0-based)')
    parser.add_argument('--all',         action='store_true',
                        help='Run all models sequentially')
    parser.add_argument('--n_restarts',  type=int, default=5)
    parser.add_argument('--n_samples',   type=int, default=None,
                        help='Cap samples per model (default: all)')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--pkl',         default=PKL)
    args = parser.parse_args()

    PKL = args.pkl

    os.makedirs(RECOVERY_DIR, exist_ok=True)
    all_models = get_all_models(open_source_only=True)
    print(f'Found {len(all_models)} open-source models with full+myopic fits')

    # Select which models to run
    if args.jobid is not None:
        if args.jobid >= len(all_models):
            print(f'jobid {args.jobid} out of range (n={len(all_models)})')
            sys.exit(1)
        targets = [all_models[args.jobid]]
    elif args.model is not None:
        targets = [args.model]
    elif args.all:
        targets = all_models
    else:
        print('Specify --model, --jobid, or --all')
        print('Available models:')
        for i, m in enumerate(all_models):
            print(f'  [{i}] {m}')
        sys.exit(0)

    for sname in targets:
        print(f'\n=== {sname} ===')
        out_path = os.path.join(RECOVERY_DIR, f'recovery_{sname}.json')
        if os.path.exists(out_path):
            print(f'  Already exists, skipping. Delete to rerun.')
            continue

        result = run_recovery_for_model(
            sname,
            n_restarts=args.n_restarts,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        if result is None:
            continue

        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  Saved to {out_path}')

    # Aggregate if all done
    if args.all or args.jobid is not None:
        _aggregate()


def _aggregate():
    files = glob.glob(os.path.join(RECOVERY_DIR, 'recovery_*.json'))
    if not files:
        return
    all_results = []
    for f in sorted(files):
        with open(f) as fp:
            all_results.append(json.load(fp))
    out = os.path.join(RESULTS_DIR, 'model_recovery_all.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nAggregated {len(all_results)} models → {out}')

    # Print summary
    print(f'\n{"Model":<45} {"SimFull Δ":>10} {"SimMyopic Δ":>12} {"Recovery":>10}')
    print('-' * 80)
    correct = 0
    for r in all_results:
        sf = r['sim_full']['delta']      # should be > 0
        sm = r['sim_myopic']['delta']    # should be < 0
        ok = sf > 0 and sm < 0
        if ok:
            correct += 1
        print(f'{r["sname"]:<45} {sf:>+10.4f} {sm:>+12.4f} {"✓" if ok else "✗":>10}')
    print(f'\nRecovery rate: {correct}/{len(all_results)}')


if __name__ == '__main__':
    main()
