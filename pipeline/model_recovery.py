"""
Model recovery for the full-tree vs myopic comparison (Appendix D.2).

For one LLM, simulate choices on its own extracted trees from (1) the fitted
full-tree model and (2) the fitted myopic model, then refit both models to each
synthetic set. delta = (NLL_myopic - NLL_full) / N should be > 0 for (1) and < 0
for (2). Requires results/fits/pattern/<model>_{full,myopic}.json.

    python pipeline/model_recovery.py --model deepseek-r1
    python pipeline/model_recovery.py --jobid $SLURM_ARRAY_TASK_ID

Writes results/recovery/<model>.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fiar.fit import build_dataset, fit_model, load_turns, make_myopic, sample_choices
from fiar.models import FITTED_MODELS


def refit_both(dataset, restarts, seed):
    """Fit full-tree and myopic models to the same choices; return per-turn NLLs and delta."""
    n = len(dataset)
    full = fit_model(dataset, n_restarts=restarts, seed=seed)['nll'] / n
    myopic = fit_model(make_myopic(dataset), n_restarts=restarts, seed=seed)['nll'] / n
    return {'nll_full': full, 'nll_myopic': myopic, 'delta': myopic - full}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', help='paper model name, e.g. deepseek-r1')
    ap.add_argument('--jobid', type=int, help=f'index into FITTED_MODELS (0-{len(FITTED_MODELS) - 1})')
    ap.add_argument('--restarts', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pkl', default='data/game_trees.pkl')
    ap.add_argument('--fitdir', default='results/fits/pattern')
    ap.add_argument('--outdir', default='results/recovery')
    args = ap.parse_args()

    model = args.model or FITTED_MODELS[args.jobid]
    fits = {v: json.load(open(Path(args.fitdir) / f'{model}_{v}.json')) for v in ('full', 'myopic')}
    dataset = build_dataset(load_turns(args.pkl, model))
    rng = np.random.default_rng(args.seed)

    def simulate(data, fit):
        return sample_choices(data, 'pattern', np.array(fit['weights']), fit['C'], 1.0, rng)

    sim_full = simulate(dataset, fits['full'])
    # Myopic choices are simulated on depth-1 trees, then paired with the full trees for refitting.
    sim_myopic = [{**s, 'observed_move': m['observed_move']}
                  for s, m in zip(dataset, simulate(make_myopic(dataset), fits['myopic']))]

    result = {'model': model, 'n_samples': len(dataset),
              'sim_full': refit_both(sim_full, args.restarts, args.seed),
              'sim_myopic': refit_both(sim_myopic, args.restarts, args.seed)}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f'{model}.json').write_text(json.dumps(result, indent=2))
    print(f"{model}: delta(sim full) = {result['sim_full']['delta']:+.4f}, "
          f"delta(sim myopic) = {result['sim_myopic']['delta']:+.4f}")


if __name__ == '__main__':
    main()
