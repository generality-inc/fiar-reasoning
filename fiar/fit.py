"""
Maximum-likelihood fitting of the cognitive models (paper Section 4.1, Appendix C.4).

Move choice is a softmax over the backed-up values of the candidate depth-1
moves (Eq. 4). Free parameters are the feature weights, the active scaling
constant C (pattern basis only) and, for the discount model, gamma.
Parameters are optimised with L-BFGS-B from several random starts.
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_softmax, logit, softmax

from .board import COLS, EMPTY, ROWS, apply_move, parse_board_state, parse_move, parse_tree
from .features import BASES
from .minimax import annotate_tree_with_boards, root_action_values

VARIANTS = ('full', 'myopic', 'notree', 'gamma')
C_BOUNDS = (math.log(0.25), math.log(5.0))


# --- data ------------------------------------------------------------------

def load_turns(pkl_path: str, model: str) -> pd.DataFrame:
    """Turns of one LLM with a legal move, a board state and a non-empty extracted tree."""
    df = pd.read_pickle(pkl_path)
    df = df[(df['model_names'] == model) & df['move_success']]
    df = df[df['board_state'].notna() & (df['board_state'].str.strip() != '')]
    df = df[df['extracted_move'].map(parse_move).notna()]
    return df[df['trees'].map(lambda t: isinstance(t, list) and len(t) > 0)]


def build_dataset(df: pd.DataFrame) -> list:
    """
    Turn rows into samples {'root_board', 'tree', 'observed_move'}, keeping turns
    with at least two candidate moves, all moves on the board, and the chosen move
    among the (legal) candidates. Trees are annotated with board states once.
    """
    def all_moves(nodes):
        for node in nodes:
            yield node['move']
            yield from all_moves(node['children'])

    dataset = []
    for _, row in df.iterrows():
        try:
            _, _, root_board = parse_board_state(row['board_state'])
            tree = parse_tree(row['trees'])
        except (ValueError, TypeError):
            continue
        move = parse_move(row['extracted_move'])
        if len(tree) < 2 or any(not (0 <= r < ROWS and 0 <= c < COLS) for r, c in all_moves(tree)):
            continue
        if move not in [node['move'] for node in tree]:
            continue
        annotate_tree_with_boards(tree, root_board)
        if next(n for n in tree if n['move'] == move)['board_self'] is None:
            continue
        dataset.append({'root_board': root_board, 'tree': tree, 'observed_move': move})
    return dataset


def make_myopic(dataset: list) -> list:
    """Keep only the depth-1 nodes of each tree."""
    return [{**s, 'tree': [{'move': n['move'], 'children': [], 'board_self': n['board_self']}
                           for n in s['tree']]}
            for s in dataset]


def make_notree(dataset: list) -> list:
    """Replace each tree with all legal moves as depth-1 leaves."""
    out = []
    for s in dataset:
        board = s['root_board']
        tree = [{'move': (r, c), 'children': [], 'board_self': apply_move(board, r, c)}
                for r in range(ROWS) for c in range(COLS) if board[r, c] == EMPTY]
        out.append({**s, 'tree': tree})
    return out


def variant_dataset(dataset: list, variant: str) -> list:
    if variant == 'myopic':
        return make_myopic(dataset)
    if variant == 'notree':
        return make_notree(dataset)
    return dataset


# --- likelihood ------------------------------------------------------------

def _choice_values(sample, basis, weights, C, gamma):
    values = root_action_values(sample['tree'], BASES[basis].value, weights, C, gamma)
    moves = list(values)
    return moves, np.array([values[m] for m in moves])


def neg_log_likelihood(dataset, basis, weights, C=1.0, gamma=1.0) -> float:
    nll = 0.0
    for s in dataset:
        moves, q = _choice_values(s, basis, weights, C, gamma)
        nll -= log_softmax(q)[moves.index(s['observed_move'])]
    return nll


def accuracy(dataset, basis, weights, C=1.0, gamma=1.0) -> float:
    """Fraction of turns where the highest-valued candidate is the chosen move."""
    hits = 0
    for s in dataset:
        moves, q = _choice_values(s, basis, weights, C, gamma)
        hits += moves[int(np.argmax(q))] == s['observed_move']
    return hits / len(dataset)


def sample_choices(dataset, basis, weights, C, gamma, rng) -> list:
    """Replace each observed move with one drawn from the model's softmax policy."""
    out = []
    for s in dataset:
        moves, q = _choice_values(s, basis, weights, C, gamma)
        out.append({**s, 'observed_move': moves[rng.choice(len(moves), p=softmax(q))]})
    return out


# --- fitting ---------------------------------------------------------------

def fit_model(dataset, basis='pattern', fit_C=True, fit_gamma=False,
              n_restarts=20, seed=42, warm_starts=()) -> dict:
    """
    Minimise the NLL over the feature weights, plus C if fit_C (else C = 1) and
    gamma if fit_gamma (else gamma = 1). C is fitted as log C within C_BOUNDS and
    gamma as logit(gamma).

    warm_starts: extra starting points, each a dict with 'weights' and optionally
    'C' / 'gamma' (natural units), run before the random restarts.
    """
    spec = BASES[basis]
    n_w = len(spec.feature_names)
    win = spec.feature_names.index(spec.win_feature)
    rng = np.random.default_rng(seed)

    def unpack(x):
        C = math.exp(x[n_w]) if fit_C else 1.0
        gamma = expit(x[-1]) if fit_gamma else 1.0
        return x[:n_w], C, gamma

    def objective(x):
        return neg_log_likelihood(dataset, basis, *unpack(x))

    bounds = [(None, None)] * n_w + [C_BOUNDS] * fit_C + [(None, None)] * fit_gamma

    starts = []
    for ws in warm_starts:
        x0 = list(ws['weights'])
        if fit_C:
            x0.append(math.log(ws.get('C', spec.init_C)))
        if fit_gamma:
            x0.append(logit(np.clip(ws.get('gamma', 0.5), 1e-4, 1 - 1e-4)))
        starts.append(np.array(x0))
    for _ in range(n_restarts):
        x0 = [abs(rng.normal(2.0, 0.5)) if i == win else rng.normal(0, 0.5) for i in range(n_w)]
        if fit_C:
            x0.append(rng.normal(math.log(spec.init_C), 0.3))
        if fit_gamma:
            x0.append(rng.normal(logit(0.5), 0.5))
        starts.append(np.array(x0))

    best = None
    for i, x0 in enumerate(starts):
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6})
        print(f'  start {i + 1}/{len(starts)}: NLL={res.fun:.4f}')
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError('All restarts failed.')

    weights, C, gamma = unpack(best.x)
    return {'weights': np.asarray(weights), 'C': C, 'gamma': gamma,
            'nll': float(best.fun), 'n_samples': len(dataset)}
