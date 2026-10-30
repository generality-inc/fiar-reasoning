"""
Fit one cognitive model variant to one LLM's move choices (paper Section 4, Appendix D.4).

Variants: full (minimax over the extracted tree), myopic (depth-1 nodes only),
notree (all legal moves), gamma (discount model, gamma fitted).
Bases: pattern (main text) or windowcount (Appendix D.4; C fixed to 1, and the
discount fit is also started from the myopic solution, so run --variant myopic first).

    python pipeline/fit_cognitive_models.py --model deepseek-r1 --variant myopic
    python pipeline/fit_cognitive_models.py --jobid $SLURM_ARRAY_TASK_ID --variant gamma --basis windowcount

Writes results/fits/<basis>/<model>_<variant>.json.
"""

import argparse
import json
from pathlib import Path

from fiar.fit import VARIANTS, accuracy, build_dataset, fit_model, load_turns, variant_dataset
from fiar.models import FITTED_MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', help='paper model name, e.g. deepseek-r1')
    ap.add_argument('--jobid', type=int, help=f'index into FITTED_MODELS (0-{len(FITTED_MODELS) - 1})')
    ap.add_argument('--variant', choices=VARIANTS, default='full')
    ap.add_argument('--basis', choices=['pattern', 'windowcount'], default='pattern')
    ap.add_argument('--restarts', type=int, default=20)
    ap.add_argument('--pkl', default='data/game_trees.pkl')
    ap.add_argument('--outdir', default='results/fits')
    args = ap.parse_args()

    model = args.model or FITTED_MODELS[args.jobid]
    outdir = Path(args.outdir) / args.basis
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = variant_dataset(build_dataset(load_turns(args.pkl, model)), args.variant)
    print(f'{model} | {args.basis} | {args.variant} | {len(dataset)} turns')

    fit_C = args.basis == 'pattern'
    fit_gamma = args.variant == 'gamma'
    warm_starts = []
    if fit_gamma and args.basis == 'windowcount':
        myopic = json.load(open(outdir / f'{model}_myopic.json'))
        warm_starts = [{'weights': myopic['weights'], 'gamma': 1e-4}]

    res = fit_model(dataset, args.basis, fit_C=fit_C, fit_gamma=fit_gamma,
                    n_restarts=args.restarts, warm_starts=warm_starts)
    summary = {
        'model': model,
        'basis': args.basis,
        'variant': args.variant,
        'n_samples': len(dataset),
        'weights': res['weights'].tolist(),
        'C': res['C'],
        'gamma': res['gamma'],
        'nll_per_sample': res['nll'] / len(dataset),
        'accuracy': accuracy(dataset, args.basis, res['weights'], res['C'], res['gamma']),
    }
    path = outdir / f'{model}_{args.variant}.json'
    path.write_text(json.dumps(summary, indent=2))
    print(f"NLL/turn {summary['nll_per_sample']:.4f}, accuracy {summary['accuracy']:.3f} -> {path}")


if __name__ == '__main__':
    main()
