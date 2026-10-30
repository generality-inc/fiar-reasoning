"""
Heuristic value functions over board features.

Two feature bases are provided:

pattern (main text, Eq. 1; van Opheusden et al., 2023)
    h(s) = w_centre * (centre_self - centre_opp)
         + sum_i w_i * (C * phi_i(self) - phi_i(opp))
    with phi_i counting connected-2, unconnected-2, three and four in viable
    length-4 windows (windows containing no opponent piece).

windowcount (Appendix D.4)
    h(s) = w_0 * (centre_self - centre_opp) + sum_k w_k * n_k
    with n_k counting length-4 windows by their composition (n_self, n_opp),
    1 <= n_self + n_opp <= 4 (14 compositions). Mirror compositions have their
    own weights, so C is not used (it is fixed to 1 during fitting).

Both share the centre term (sum of inverse distances to the board centre) and
the 69 length-4 windows. Each window is encoded as a base-3 key
(empty=0, self=1, opp=2) and features are read from precomputed lookup tables.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .board import ROWS, COLS

WINDOW = 4

_centre = np.array([(ROWS - 1) / 2, (COLS - 1) / 2])
_dist = np.sqrt((np.arange(ROWS)[:, None] - _centre[0])**2 + (np.arange(COLS)[None, :] - _centre[1])**2)
CENTRE_TABLE = 1.0 / _dist


def _enumerate_windows():
    rows, cols = [], []
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        for r in range(ROWS):
            for c in range(COLS):
                squares = [(r + i * dr, c + i * dc) for i in range(WINDOW)]
                if all(0 <= sr < ROWS and 0 <= sc < COLS for sr, sc in squares):
                    rows.append([sr for sr, _ in squares])
                    cols.append([sc for _, sc in squares])
    return np.array(rows, dtype=np.intp), np.array(cols, dtype=np.intp)


_WIN_ROWS, _WIN_COLS = _enumerate_windows()          # (69, 4) each
_TERNARY = np.array([1, 3, 9, 27], dtype=np.int32)
_PATTERNS = [[(key // 3**i) % 3 for i in range(WINDOW)] for key in range(81)]


def _window_keys(board_self):
    ternary = np.where(board_self == -1, 2, board_self).astype(np.int32)
    return ternary[_WIN_ROWS, _WIN_COLS] @ _TERNARY


def _centre_diff(board_self):
    return float(np.sum(CENTRE_TABLE[board_self == 1])) - float(np.sum(CENTRE_TABLE[board_self == -1]))


# --- pattern basis ---------------------------------------------------------

def _pattern_table(own, other):
    """(81, 4) table of [connected_2, unconnected_2, three, four] for one player."""
    table = np.zeros((81, 4), dtype=np.int8)
    for key, pattern in enumerate(_PATTERNS):
        if other in pattern:
            continue                                   # blocked window
        pos = [i for i, d in enumerate(pattern) if d == own]
        if len(pos) == 4:
            table[key, 3] = 1
        elif len(pos) == 3:
            table[key, 2] = 1
        elif len(pos) == 2:
            table[key, 0 if pos[1] - pos[0] == 1 else 1] = 1
    return table


_PATTERN_SELF = _pattern_table(own=1, other=2)
_PATTERN_OPP = _pattern_table(own=2, other=1)


def pattern_features(board_self):
    """Return (phi_self, phi_opp): [centre, connected_2, unconnected_2, three, four]."""
    keys = _window_keys(board_self)
    phi_self = np.array([np.sum(CENTRE_TABLE[board_self == 1]), *_PATTERN_SELF[keys].sum(axis=0)])
    phi_opp = np.array([np.sum(CENTRE_TABLE[board_self == -1]), *_PATTERN_OPP[keys].sum(axis=0)])
    return phi_self, phi_opp


def pattern_value(board_self, weights, C=1.0):
    phi_self, phi_opp = pattern_features(board_self)
    return weights[0] * (phi_self[0] - phi_opp[0]) + float(np.dot(weights[1:], C * phi_self[1:] - phi_opp[1:]))


# --- window-count basis ----------------------------------------------------

COMPOSITIONS = [(a, n - a) for n in range(1, WINDOW + 1) for a in range(n, -1, -1)]  # (n_self, n_opp)


def _composition_table():
    table = np.zeros((81, len(COMPOSITIONS)), dtype=np.int8)
    for key, pattern in enumerate(_PATTERNS):
        comp = (pattern.count(1), pattern.count(2))
        if sum(comp):
            table[key, COMPOSITIONS.index(comp)] = 1
    return table


_COMPOSITION = _composition_table()


def windowcount_features(board_self):
    """Return [centre_self - centre_opp, n_(1,0), n_(0,1), ..., n_(0,4)] (length 15)."""
    return np.array([_centre_diff(board_self), *_COMPOSITION[_window_keys(board_self)].sum(axis=0)],
                    dtype=np.float64)


def windowcount_value(board_self, weights, C=1.0):
    return float(np.dot(weights, windowcount_features(board_self)))


# --- registry --------------------------------------------------------------

@dataclass(frozen=True)
class Basis:
    name: str
    feature_names: list
    value: Callable          # value(board_self, weights, C) -> float
    win_feature: str         # weight initialised positive (own four-in-a-row)
    init_C: float            # mean initial C when C is fitted


BASES = {
    'pattern': Basis('pattern', ['w_centre', 'w_c2', 'w_uc2', 'w_three', 'w_four'],
                     pattern_value, win_feature='w_four', init_C=1.3),
    'windowcount': Basis('windowcount', ['w_centre'] + [f'w_{a}s{b}o' for a, b in COMPOSITIONS],
                         windowcount_value, win_feature='w_4s0o', init_C=1.0),
}
