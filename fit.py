"""
fit.py
======
Maximum-likelihood estimation of the heuristic search model parameters
for a single LLM model.

Model
-----
For each turn t, the agent has a set of root actions A_t from the tree.
Each action a gets a minimax value Q_t(a; w, C, T), where T is the
terminal value assigned to a four-in-a-row position.

The probability of the observed move m_t is a softmax over Q-values:

    P(m_t | A_t, w, C, T) = exp(Q_t(m_t)) / sum_{a in A_t} exp(Q_t(a))

Parameters to fit (7 total):
    w : np.ndarray, shape (5,)  [w_centre, w_c2, w_uc2, w_three, w_four]
    C : float > 0          active scaling constant, parameterised as exp(log_C)
    lam : float in (0,1)   lapse rate, parameterised as sigmoid(logit_lam)

Pipeline
--------
1. load_and_filter()    -- load df, filter to one model, apply edge-case filters
2. build_dataset()      -- parse trees, annotate board states, collect samples
3. neg_log_likelihood() -- the objective function
4. fit_model()          -- run optimisation with multiple restarts
"""

import math
import numpy as np
import pandas as pd
import scipy.special
from scipy.optimize import minimize
from scipy.special import log_softmax
import warnings

from preprocess import parse_board_state, parse_tree
from minimax import annotate_tree_with_boards, minimax_values


# ---------------------------------------------------------------------------
# 1. Load and filter
# ---------------------------------------------------------------------------

def load_and_filter(pkl_path: str, model_name: str) -> pd.DataFrame:
    """
    Load the dataframe and filter to a single model with clean data.

    Filters applied (in order):
        1. Keep rows where model_names == model_name
        2. Keep rows where move_success == True
        3. Keep rows where board_state is a non-empty string
        4. Keep rows where extracted_move matches 'm r c'
        5. Keep rows where trees is a non-empty list

    Returns a filtered copy of the DataFrame.
    """
    import re
    df = pd.read_pickle(pkl_path)

    df = df[df['model_names'] == model_name].copy()
    print(f"After model filter: {len(df)} rows")

    df = df[df['move_success'] == True].copy()
    print(f"After move_success filter: {len(df)} rows")

    df = df[df['board_state'].notna()].copy()
    df = df[df['board_state'].str.strip() != ''].copy()
    print(f"After board_state filter: {len(df)} rows")

    move_ok = df['extracted_move'].apply(
        lambda s: bool(isinstance(s, str) and re.fullmatch(r'm\s+\d+\s+\d+', s.strip()))
    )
    df = df[move_ok].copy()
    print(f"After valid move filter: {len(df)} rows")

    trees_ok = df['trees'].apply(lambda t: isinstance(t, list) and len(t) > 0)
    df = df[trees_ok].copy()
    print(f"After non-empty trees filter: {len(df)} rows")

    return df


# ---------------------------------------------------------------------------
# 2. Build dataset
# ---------------------------------------------------------------------------

def build_dataset(df: pd.DataFrame, trees_col: str = 'trees') -> list:
    """
    Parse and preprocess every row into a dataset list.

    Each element is a dict:
        {
          'root_board'   : np.ndarray (4,9)  -- board before any move in tree
          'tree'         : list of parsed+annotated node dicts
          'observed_move': (row, col)
          'game_path'    : str  (for debugging)
          'turn_idx'     : int
        }

    Edge cases handled:
        - Skip if board_state or tree fails to parse
        - Skip if tree has fewer than 2 root actions (no meaningful choice)
        - Skip if observed move is not among the root nodes
        - Skip if the observed move's node has board_self=None (occupied square)

    Returns list of valid samples.
    """
    import re
    dataset = []
    skipped = {'parse_error': 0, 'too_few_roots': 0,
               'out_of_bounds': 0, 'move_not_in_tree': 0, 'invalid_node': 0}

    for _, row in df.iterrows():
        # --- Parse board state ---
        try:
            _, _, root_board = parse_board_state(row['board_state'])
        except Exception:
            skipped['parse_error'] += 1
            continue

        # --- Parse tree ---
        try:
            tree = parse_tree(row[trees_col])
        except Exception:
            skipped['parse_error'] += 1
            continue

        # --- Need at least 2 root actions to form a meaningful choice ---
        if len(tree) < 2:
            skipped['too_few_roots'] += 1
            continue

        # --- Parse observed move ---
        m = re.fullmatch(r'm\s+(\d+)\s+(\d+)', row['extracted_move'].strip())
        observed_move = (int(m.group(1)), int(m.group(2)))

        # --- All moves in the tree must be within board bounds ---
        # LLMs occasionally generate moves like (4, x) which are off the
        # 4x9 board. Check every node recursively before annotating.
        def _all_moves(nodes):
            for node in nodes:
                yield node['move']
                yield from _all_moves(node.get('children', []))

        if any(not (0 <= r < 4 and 0 <= c < 9)
               for r, c in _all_moves(tree)):
            skipped['out_of_bounds'] += 1
            continue

        # --- Observed move must be one of the root nodes ---
        root_moves = [node['move'] for node in tree]
        if observed_move not in root_moves:
            skipped['move_not_in_tree'] += 1
            continue

        # --- Annotate every node with its board state (done once, cached) ---
        annotate_tree_with_boards(tree, root_board)

        # --- Observed move's node must have a valid board ---
        obs_node = next(n for n in tree if n['move'] == observed_move)
        if obs_node.get('board_self') is None:
            skipped['invalid_node'] += 1
            continue

        dataset.append({
            'root_board':    root_board,
            'tree':          tree,
            'observed_move': observed_move,
            'game_path':     row.get('game_path', ''),
            'turn_idx':      row.get('turn_idx', -1),
        })

    print(f"\nDataset built: {len(dataset)} valid samples")
    print(f"Skipped: {skipped}")
    return dataset


# ---------------------------------------------------------------------------
# 3. Negative log-likelihood
# ---------------------------------------------------------------------------

def neg_log_likelihood(params: np.ndarray, dataset: list) -> float:
    """
    Compute the negative log-likelihood over the dataset.

    Parameters
    ----------
    params : np.ndarray, shape (8,)
        [w_centre, w_c2, w_uc2, w_three, w_four, log_C, logit_lam, logit_gamma]

    C     is parameterised as exp(log_C)        to enforce C > 0.
    lam   is parameterised as sigmoid(logit_lam) to keep it in (0, 1).
    gamma is parameterised as sigmoid(logit_gamma) to keep it in (0, 1).
    """
    w     = params[:5]                              # [w_centre, w_c2, w_uc2, w_three, w_four]
    C     = np.exp(params[5])                       # active scaling constant
    lam   = 1.0 / (1.0 + np.exp(-params[6]))       # sigmoid: lapse rate in (0, 1)
    gamma = 1.0 / (1.0 + np.exp(-params[7]))       # sigmoid: minimax discount in (0, 1)

    nll = 0.0
    for sample in dataset:
        action_values = minimax_values(sample['tree'], w, C, gamma)

        moves  = list(action_values.keys())
        qvals  = np.array([action_values[mv] for mv in moves])
        n      = len(moves)

        # Mixture: (1-lam) * softmax(Q) + lam * uniform
        probs     = (1.0 - lam) * np.exp(log_softmax(qvals)) + lam / n
        obs_idx   = moves.index(sample['observed_move'])
        nll -= np.log(probs[obs_idx])

    return nll


# ---------------------------------------------------------------------------
# 4. Fit
# ---------------------------------------------------------------------------

def fit_model(dataset: list,
              n_restarts: int = 5,
              seed: int = 42,
              fixed_params: dict = None) -> dict:
    """
    Fit model parameters by minimizing the negative log-likelihood.

    Uses L-BFGS-B with multiple random restarts to avoid local minima.

    Parameters
    ----------
    dataset      : list from build_dataset()
    n_restarts   : number of random initial conditions
    seed         : random seed
    fixed_params : dict of parameters to hold fixed, e.g.
                       {'C': 1.0, 'lam': 0.0}
                   Any subset of the 7 parameters can be fixed:
                       'w_centre', 'w_c2', 'w_uc2', 'w_three', 'w_four',
                       'C', 'lam'
                   Fixed parameters are excluded from optimisation.
                   Free parameters are still randomly initialised.

    Returns
    -------
    dict with keys:
        'weights'      : np.ndarray (5,)  [w_centre, w_c2, w_uc2, w_three, w_four]
        'C'            : float            active scaling constant
        'lam'          : float            lapse rate (in (0,1))
        'nll'          : float            best negative log-likelihood
        'nll_chance'   : float            NLL of uniform (chance) model
        'n_params'     : int              number of FREE parameters
        'n_samples'    : int              number of turns fit
        'fixed_params' : dict             which parameters were fixed and at what value
        'all_results'  : list             raw scipy results from all restarts
    """
    rng = np.random.default_rng(seed)
    if fixed_params is None:
        fixed_params = {}

    # Full parameter names and their indices in the 8-vector
    PARAM_NAMES = ['w_centre', 'w_c2', 'w_uc2', 'w_three', 'w_four', 'C', 'lam', 'gamma']
    # Default initialisations for each parameter (in natural units)
    DEFAULTS = {
        'w_centre': 0.0, 'w_c2': 0.0, 'w_uc2': 0.0,
        'w_three': 0.0,  'w_four': 2.0,
        'C': 1.3,        'lam': 0.05,  'gamma': 0.5,
    }
    # Bounds in unconstrained space for each parameter
    BOUNDS = {
        'w_centre': (None, None), 'w_c2': (None, None),
        'w_uc2':    (None, None), 'w_three': (None, None),
        'w_four':   (None, None),
        'C':        (math.log(0.25), math.log(5.0)),
        'lam':      (scipy.special.logit(0.001), None),
        'gamma':    (None, None),
    }

    # Convert fixed_params from natural units to unconstrained space
    def to_unconstrained(name, val):
        if name == 'C':     return math.log(val)
        if name == 'lam':   return scipy.special.logit(val)
        if name == 'gamma': return scipy.special.logit(val)
        return val  # weights are unconstrained

    def to_natural(name, val):
        if name == 'C':     return float(np.exp(val))
        if name == 'lam':   return float(1.0 / (1.0 + np.exp(-val)))
        if name == 'gamma': return float(1.0 / (1.0 + np.exp(-val)))
        return float(val)

    fixed_unconstrained = {name: to_unconstrained(name, val)
                           for name, val in fixed_params.items()}

    # Which parameters are free (to be optimised)?
    free_names = [name for name in PARAM_NAMES if name not in fixed_params]
    n_free = len(free_names)

    nll_chance = sum(np.log(len(s['tree'])) for s in dataset)

    # Wrapper: takes free-parameter vector, inserts fixed values, calls NLL
    def _objective(free_vals):
        full = np.zeros(8)
        free_idx = 0
        for i, name in enumerate(PARAM_NAMES):
            if name in fixed_unconstrained:
                full[i] = fixed_unconstrained[name]
            else:
                full[i] = free_vals[free_idx]
                free_idx += 1
        return neg_log_likelihood(full, dataset)

    best_result = None
    all_results = []

    for restart in range(n_restarts):
        # Random initialisation for free parameters only
        x0 = []
        for name in free_names:
            if name == 'w_four':
                x0.append(abs(rng.normal(2.0, 0.5)))
            elif name == 'C':
                x0.append(rng.normal(math.log(DEFAULTS['C']), 0.3))
            elif name == 'lam':
                x0.append(rng.normal(scipy.special.logit(0.05), 0.5))  # lam ~ 0.05
            elif name == 'gamma':
                x0.append(rng.normal(scipy.special.logit(0.5), 0.5))   # gamma ~ 0.5
            else:
                x0.append(rng.normal(0, 0.5))
        x0 = np.array(x0)

        bounds = [BOUNDS[name] for name in free_names]

        try:
            result = minimize(
                fun=_objective,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6},
            )
        except Exception as e:
            warnings.warn(f"Restart {restart + 1} failed: {e}")
            continue

        if np.isnan(result.fun):
            print(f"  Restart {restart+1}/{n_restarts}: NLL=NaN (skipped)")
            continue

        all_results.append(result)
        if best_result is None or result.fun < best_result.fun:
            best_result = result

        print(f"  Restart {restart+1}/{n_restarts}: NLL={result.fun:.4f} "
              f"({'converged' if result.success else 'not converged'})")

    if best_result is None:
        raise RuntimeError("All restarts failed.")

    # Reconstruct full parameter vector from best free params + fixed values
    full = np.zeros(8)
    free_idx = 0
    for i, name in enumerate(PARAM_NAMES):
        if name in fixed_unconstrained:
            full[i] = fixed_unconstrained[name]
        else:
            full[i] = best_result.x[free_idx]
            free_idx += 1

    return {
        'weights':      full[:5],
        'C':            to_natural('C',     full[5]),
        'lam':          to_natural('lam',   full[6]),
        'gamma':        to_natural('gamma', full[7]),
        'nll':          float(best_result.fun),
        'nll_chance':   float(nll_chance),
        'n_params':     n_free,
        'n_samples':    len(dataset),
        'dataset':      dataset,
        'fixed_params': fixed_params,
        'all_results':  all_results,
    }


def compute_accuracy(dataset: list, weights: np.ndarray,
                     C: float, lam: float, gamma: float = 1.0) -> float:
    """
    Fraction of turns where the model's predicted move matches the observed move.

    The predicted move is the one with highest probability under the mixture
    model: (1-lam)*softmax(Q) + lam/n. Since lam/n is uniform, the ranking
    is identical to ranking by Q-value alone, so this is equivalent to
    predicting argmax Q.

    Parameters
    ----------
    dataset : list from build_dataset()
    weights : np.ndarray, shape (5,)
    C       : float  active scaling constant
    lam     : float  lapse rate in (0, 1)

    Returns
    -------
    float  accuracy in [0, 1]
    """
    correct = 0
    for sample in dataset:
        action_values = minimax_values(sample['tree'], weights, C, gamma)
        moves  = list(action_values.keys())
        qvals  = np.array([action_values[mv] for mv in moves])
        pred   = moves[int(np.argmax(qvals))]
        if pred == sample['observed_move']:
            correct += 1
    return correct / len(dataset)


def print_results(results: dict) -> None:
    """Pretty-print the fitting results."""
    w     = results['weights']
    names = ['centre', 'connected_2', 'unconnected_2', 'three', 'four']
    fixed = results.get('fixed_params', {})

    print("\n=== Fitted Parameters ===")
    param_map = {'w_centre': w[0], 'w_c2': w[1], 'w_uc2': w[2],
                 'w_three': w[3], 'w_four': w[4]}
    for name, val in zip(names, w):
        tag = '  (fixed)' if f'w_{name}' in fixed else ''
        print(f"  w_{name:<15s} = {val:+.4f}{tag}")
    tag_C   = '  (fixed)' if 'C'   in fixed else ''
    tag_lam = '  (fixed)' if 'lam' in fixed else ''
    print(f"  C (active scale)  = {results['C']:.4f}{tag_C}")
    print(f"  lam (lapse rate)  = {results['lam']:.4f}{tag_lam}")
    print(f"  Free parameters   = {results['n_params']}")
    print(f"\n=== Fit Quality ===")
    print(f"  N turns           = {results['n_samples']}")
    print(f"  NLL (model)       = {results['nll']:.4f}")
    print(f"  NLL (chance)      = {results['nll_chance']:.4f}")
    nll_per    = results['nll']        / results['n_samples']
    chance_per = results['nll_chance'] / results['n_samples']
    print(f"  NLL/turn (model)  = {nll_per:.4f}   (chance = {chance_per:.4f})")
    delta = results['nll_chance'] - results['nll']
    print(f"  ΔLL vs chance     = {delta:.4f}  "
          f"({'better' if delta > 0 else 'worse'} than chance)")
    acc = compute_accuracy(results['dataset'], results['weights'],
                           results['C'], results['lam'])
    chance_acc = np.mean([1.0 / (s['root_board'] == 0).sum() for s in results['dataset']])
    print(f"  Accuracy          = {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Chance accuracy   = {chance_acc:.4f}  ({chance_acc*100:.1f}%)")

    
# ---------------------------------------------------------------------------
# 5. Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(dataset: list,
                   n_folds: int = 5,
                   n_restarts: int = 5,
                   seed: int = 42,
                   fixed_params: dict = None) -> dict:
    """
    5-fold cross-validation: fit on 4 folds, evaluate on the held-out fold.

    For each fold:
      - Parameters are fitted on the training set (4/5 of the data)
      - NLL and accuracy are evaluated on the test set (1/5 of the data)

    Parameters
    ----------
    dataset      : full dataset from build_dataset()
    n_folds      : number of folds (default 5)
    n_restarts   : restarts per fold (passed to fit_model)
    seed         : random seed for fold assignment and optimisation
    fixed_params : passed through to fit_model

    Returns
    -------
    dict with keys:
        'fold_results'   : list of per-fold dicts, each containing
                           'train_nll', 'test_nll', 'test_nll_chance',
                           'test_accuracy', 'test_chance_accuracy', 'weights',
                           'C', 'lam', 'n_train', 'n_test'
        'mean_test_nll'          : float  mean NLL per sample on test folds
        'mean_test_accuracy'     : float  mean accuracy on test folds
        'mean_test_chance_acc'   : float  mean chance accuracy on test folds
        'mean_test_nll_chance'   : float  mean chance NLL per sample on test folds
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    fold_results = []

    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])
        train_set = [dataset[i] for i in train_idx]
        test_set  = [dataset[i] for i in test_idx]

        print(f"\n--- Fold {fold_idx+1}/{n_folds}: "
              f"{len(train_set)} train / {len(test_set)} test ---")

        # Fit on training set
        results = fit_model(train_set,
                            n_restarts=n_restarts,
                            seed=seed + fold_idx,
                            fixed_params=fixed_params)

        w     = results['weights']
        C     = results['C']
        lam   = results['lam']
        gamma = results.get('gamma', 1.0)

        # Evaluate on test set
        logit_gamma = scipy.special.logit(np.clip(gamma, 1e-6, 1 - 1e-6))
        test_nll        = neg_log_likelihood(
            np.concatenate([w, [np.log(C),
                                np.log(lam / (1 - lam)),
                                logit_gamma]]),
            test_set)
        test_nll_chance = sum(np.log(len(s['tree'])) for s in test_set)
        test_acc        = compute_accuracy(test_set, w, C, lam, gamma)
        test_chance_acc = float(np.mean([1.0 / (s['root_board'] == 0).sum() for s in test_set]))

        fold_result = {
            'fold':               fold_idx + 1,
            'n_train':            len(train_set),
            'n_test':             len(test_set),
            'train_nll':          results['nll'],
            'test_nll':           float(test_nll),
            'test_nll_chance':    float(test_nll_chance),
            'test_nll_per_sample':       float(test_nll) / len(test_set),
            'test_nll_chance_per_sample': float(test_nll_chance) / len(test_set),
            'test_accuracy':      test_acc,
            'test_chance_accuracy': test_chance_acc,
            'weights':            w,
            'C':                  C,
            'lam':                lam,
        }
        fold_results.append(fold_result)

        print(f"  Test NLL/sample : {fold_result['test_nll_per_sample']:.4f}  "
              f"(chance = {fold_result['test_nll_chance_per_sample']:.4f})")
        print(f"  Test accuracy   : {test_acc:.4f}  "
              f"(chance = {test_chance_acc:.4f})")

    mean_test_nll       = float(np.mean([f['test_nll_per_sample'] for f in fold_results]))
    mean_test_acc       = float(np.mean([f['test_accuracy']       for f in fold_results]))
    mean_test_chance    = float(np.mean([f['test_nll_chance_per_sample'] for f in fold_results]))
    mean_test_chance_acc = float(np.mean([f['test_chance_accuracy'] for f in fold_results]))

    print(f"\n=== Cross-validation summary ({n_folds} folds) ===")
    print(f"  Mean test NLL/sample : {mean_test_nll:.4f}  (chance = {mean_test_chance:.4f})")
    print(f"  Mean test accuracy   : {mean_test_acc:.4f}  (chance = {mean_test_chance_acc:.4f})")

    return {
        'fold_results':           fold_results,
        'mean_test_nll':          mean_test_nll,
        'mean_test_accuracy':     mean_test_acc,
        'mean_test_nll_chance':   mean_test_chance,
        'mean_test_chance_acc':   mean_test_chance_acc,
    }


