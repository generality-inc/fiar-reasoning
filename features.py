"""
features.py
===========
Feature computation for the Four-in-a-Row heuristic value function,
following van Opheusden et al. (2023) exactly.

Board encoding (from preprocess.py)
-------------------------------------
    +1  = current player's piece (SELF)
    -1  = opponent's piece       (OPP)
     0  = empty

Value function (Eq. 1 / Eq. 3 in paper)
-----------------------------------------
    V(s) = w_centre * (centre_self - centre_opp)
         + C * dot(w[1:], phi_self[1:])      # active player's features
         -     dot(w[1:], phi_opp[1:])       # opponent's features

Features
--------
All pattern features are counted over windows of exactly 4 contiguous squares
in each of four orientations: horizontal, vertical, diagonal ↘, diagonal ↗.

Within a window, a feature for player P fires when:
  - The required P-pieces are present in the required configuration
  - ALL remaining squares in the window are EMPTY (no opponent pieces)
  This "viable window" constraint ensures the pattern can still become 4-in-a-row.

Feature definitions (all within a 4-square window):
  [0] centre       : sum of 1/dist_to_centre for P's pieces  (not a window feature)
  [1] connected_2  : exactly 2 adjacent P-pieces, other 2 squares empty
  [2] unconnected_2: exactly 2 non-adjacent P-pieces, other 2 squares empty
  [3] three        : exactly 3 P-pieces, other 1 square empty
  [4] four         : all 4 squares are P-pieces

Performance notes
-----------------
Three things are precomputed at module load and reused across all calls:

  CENTRE_TABLE  : (4, 9) float array — 1/dist_to_centre for every square.
                  Centre sums reduce to two masked np.sum calls.

  _WIN_ROWS/_WIN_COLS : (N_WINDOWS=69, 4) index arrays — all valid windows.
                  All 69 windows are extracted in one numpy slice.

  _WINDOW_FEATURES_{SELF,OPP} : (81, 4) lookup tables keyed by a ternary
                  encoding of each window (0=empty, 1=self/opp, 2=opponent/self).
                  Built once by iterating all 81 patterns; at runtime a single
                  dot product produces 69 keys and two table lookups replace all
                  per-window Python logic.
"""

import numpy as np

ROWS   = 4
COLS   = 9
WINDOW = 4   # all patterns defined over 4-contiguous-square windows


# ---------------------------------------------------------------------------
# Precomputed: centre distance lookup table
# ---------------------------------------------------------------------------
# CENTRE_TABLE[r, c] = 1 / Euclidean_distance((r,c) -> board_centre)
# Board centre is at ((ROWS-1)/2, (COLS-1)/2) = (1.5, 4.0).
# No square on the 4×9 board sits exactly at the centre, so no zero-division.

_board_centre = np.array([(ROWS - 1) / 2.0, (COLS - 1) / 2.0])
_row_coords   = np.arange(ROWS)[:, None]   # (4, 1)
_col_coords   = np.arange(COLS)[None, :]   # (1, 9)
_distances    = np.sqrt((_row_coords - _board_centre[0])**2 +
                        (_col_coords - _board_centre[1])**2)
CENTRE_TABLE  = 1.0 / _distances           # (4, 9)


# ---------------------------------------------------------------------------
# Precomputed: all window index arrays
# ---------------------------------------------------------------------------
# _WIN_ROWS[w, i] and _WIN_COLS[w, i] give the (row, col) of the i-th square
# in the w-th window.  With these, board[_WIN_ROWS, _WIN_COLS] extracts all
# 69 windows at once in a single numpy advanced-index operation.

def _enumerate_windows():
    """
    Return (row_indices, col_indices), each shape (N_WINDOWS, 4),
    for all valid 4-square windows in the four orientations.
    """
    directions = [
        (0,  1),   # horizontal
        (1,  0),   # vertical
        (1,  1),   # diagonal ↘
        (1, -1),   # diagonal ↗
    ]
    row_lists, col_lists = [], []
    for dr, dc in directions:
        for r in range(ROWS):
            for c in range(COLS):
                squares = [(r + i*dr, c + i*dc) for i in range(WINDOW)]
                if all(0 <= sr < ROWS and 0 <= sc < COLS for sr, sc in squares):
                    row_lists.append([sr for sr, sc in squares])
                    col_lists.append([sc for sr, sc in squares])
    return (np.array(row_lists, dtype=np.intp),
            np.array(col_lists, dtype=np.intp))

_WIN_ROWS, _WIN_COLS = _enumerate_windows()
N_WINDOWS = len(_WIN_ROWS)   # 69 windows on a 4×9 board


# ---------------------------------------------------------------------------
# Precomputed: per-window feature lookup tables
# ---------------------------------------------------------------------------
# We encode each window as a base-3 number: empty=0, target_player=1, opponent=2.
# There are 3^4 = 81 possible patterns.  For each, we precompute which features
# fire for the target player (self or opp).
#
# _WINDOW_FEATURES_SELF[key] = [c2, uc2, three, four]  for SELF
# _WINDOW_FEATURES_OPP [key] = [c2, uc2, three, four]  for OPP
#
# At runtime: encode board → compute 69 keys in one dot product → index table.

_TERNARY_POWERS = np.array([1, 3, 9, 27], dtype=np.int32)   # 3^0 .. 3^3


def _build_window_feature_table(target_digit: int, opponent_digit: int):
    """
    Build an (81, 4) int8 lookup table for one player perspective.

    target_digit   : ternary digit that represents 'this player's piece' (1 or 2)
    opponent_digit : ternary digit that represents 'opponent's piece'    (2 or 1)

    Returns
    -------
    table : np.ndarray, shape (81, 4), dtype int8
        Columns: [connected_2, unconnected_2, three, four]
    """
    table = np.zeros((81, 4), dtype=np.int8)

    for key in range(81):
        # Decode ternary key into a list of 4 digits
        pattern = [(key // (3**i)) % 3 for i in range(WINDOW)]

        # Viable window: opponent must not appear
        if opponent_digit in pattern:
            continue   # blocked — all features stay 0

        n_pieces = pattern.count(target_digit)

        if n_pieces == 4:
            table[key, 3] = 1   # four-in-a-row

        elif n_pieces == 3:
            table[key, 2] = 1   # three-in-a-row

        elif n_pieces == 2:
            # Positions of the two pieces within the window (0-3)
            positions = [i for i, d in enumerate(pattern) if d == target_digit]
            gap = positions[1] - positions[0]
            if gap == 1:
                table[key, 0] = 1   # connected_2  (adjacent)
            else:
                table[key, 1] = 1   # unconnected_2 (gap > 1)

        # n_pieces == 0 or 1: no feature fires

    return table


# Build both tables once:  SELF sees 1=self, 2=opp;  OPP sees 1=opp, 2=self
_WINDOW_FEATURES_SELF = _build_window_feature_table(target_digit=1, opponent_digit=2)
_WINDOW_FEATURES_OPP  = _build_window_feature_table(target_digit=2, opponent_digit=1)


# ---------------------------------------------------------------------------
# Core: vectorized feature extraction
# ---------------------------------------------------------------------------

def compute_feature_vector(board_self: np.ndarray):
    """
    Compute the full feature vector for both players in one vectorized pass.

    Parameters
    ----------
    board_self : np.ndarray, shape (4, 9), values in {-1, 0, +1}
        +1 = SELF, -1 = OPP, 0 = empty

    Returns
    -------
    phi_self : np.ndarray, shape (5,)
        [centre_self, connected_2, unconnected_2, three, four] for SELF
    phi_opp : np.ndarray, shape (5,)
        Same for OPP
    """
    # --- Centre feature ---
    # Sum 1/dist for each player's pieces using the precomputed table
    centre_self = float(np.sum(CENTRE_TABLE[board_self ==  1]))
    centre_opp  = float(np.sum(CENTRE_TABLE[board_self == -1]))

    # --- Pattern features ---
    # Step 1: re-encode board to ternary (empty=0, self=1, opp=2)
    board_ternary = np.where(board_self == -1, 2, board_self).astype(np.int32)

    # Step 2: extract all 69 windows and compute each window's ternary key
    #   window_vals shape: (69, 4)
    #   window_keys shape: (69,)  — each in [0, 80]
    window_vals = board_ternary[_WIN_ROWS, _WIN_COLS]
    window_keys = window_vals @ _TERNARY_POWERS

    # Step 3: look up features for every window at once, then sum across windows
    self_pattern_counts = _WINDOW_FEATURES_SELF[window_keys].sum(axis=0)  # (4,)
    opp_pattern_counts  = _WINDOW_FEATURES_OPP[window_keys].sum(axis=0)  # (4,)

    # Assemble feature vectors: [centre, c2, uc2, three, four]
    phi_self = np.array([centre_self, *self_pattern_counts], dtype=np.float64)
    phi_opp  = np.array([centre_opp,  *opp_pattern_counts],  dtype=np.float64)

    return phi_self, phi_opp


# ---------------------------------------------------------------------------
# Value function
# ---------------------------------------------------------------------------

def compute_value(board_self: np.ndarray,
                  weights: np.ndarray,
                  C: float = 1.0) -> float:
    """
    Compute the heuristic value V(s) from the current player's perspective.

    V(s) = w_centre * (centre_self - centre_opp)
         + dot(w[1:], C * phi_self[1:] - phi_opp[1:])

    Higher value = better for SELF (the player to move).

    Parameters
    ----------
    board_self : np.ndarray, shape (4, 9)
    weights    : np.ndarray, shape (5,)
        [w_centre, w_connected2, w_unconnected2, w_three, w_four]
    C : float
        Active scaling constant (C > 1 means active threats are valued more).

    Returns
    -------
    float
    """
    phi_self, phi_opp = compute_feature_vector(board_self)
    v  = weights[0] * (phi_self[0] - phi_opp[0])                       # centre
    v += float(np.dot(weights[1:], C * phi_self[1:] - phi_opp[1:]))    # patterns
    return v


# ---------------------------------------------------------------------------
# count_pattern_features — compatibility shim used by minimax.py
# ---------------------------------------------------------------------------

def count_pattern_features(board_self: np.ndarray):
    """
    Return pattern feature counts as nested dicts.
    Used by minimax.py to detect terminal (four-in-a-row) positions.

    Returns
    -------
    dict: {'self': {c2, uc2, three, four}, 'opp': {c2, uc2, three, four}}
    """
    phi_self, phi_opp = compute_feature_vector(board_self)

    def _to_dict(phi):
        return {
            'connected_2':   int(phi[1]),
            'unconnected_2': int(phi[2]),
            'three':         int(phi[3]),
            'four':          int(phi[4]),
        }

    return {'self': _to_dict(phi_self), 'opp': _to_dict(phi_opp)}







# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from preprocess import parse_board_state, board_to_str

    # ---- Test 1: empty board ----
    print("=== Test 1: Empty board ===")
    board_empty = np.zeros((ROWS, COLS), dtype=np.int8)
    phi_s, phi_o = compute_feature_vector(board_empty)
    print(f"phi_self = {phi_s}  (should all be 0)")
    print(f"phi_opp  = {phi_o}  (should all be 0)")

    # ---- Test 2: known position ----
    # FEN: 2WBBB3/4W4/5W3/9  (Black to move)
    # Row 0: ..W B B B . . .
    # Black has three in a row at (0,3),(0,4),(0,5): that's a three_in_a_row
    # for the window [(0,3),(0,4),(0,5),(0,6)] — squares 3 is occupied by
    # someone else? No, from Black's perspective (Black=SELF):
    #   (0,2)=OPP, (0,3)=SELF, (0,4)=SELF, (0,5)=SELF
    # Window (0,2)-(0,5): has OPP at (0,2), so NOT viable for SELF.
    # Window (0,3)-(0,6): SELF,SELF,SELF,EMPTY -> three_in_a_row fires!
    print("\n=== Test 2: FEN: 2WBBB3/4W4/5W3/9, Black to move ===")
    board_state = "FEN: 2WBBB3/4W4/5W3/9\n\nCurrent player: Black"
    _, player, board_self = parse_board_state(board_state)
    print(board_to_str(board_self, player))
    print()

    phi_s, phi_o = compute_feature_vector(board_self)
    counts = count_pattern_features(board_self)
    print(f"SELF (Black) counts: {counts['self']}")
    print(f"OPP  (White) counts: {counts['opp']}")
    print(f"phi_self = {phi_s}")
    print(f"phi_opp  = {phi_o}")

    # Manually verify: Black has BBB at (0,3),(0,4),(0,5)
    # Window [3,4,5,6]: S S S _ -> three fires ✓
    # Window [2,3,4,5]: O S S S -> NOT viable (OPP at pos 0) ✓ (no count for self)
    # White has W at (0,2),(1,4),(2,5)
    # No three or four for White
    assert counts['self']['three'] >= 1, "Expected three_in_a_row for Black"
    print("\n✓ three_in_a_row correctly detected for Black")

    # ---- Test 3: four in a row ----
    print("\n=== Test 3: Four-in-a-row ===")
    board_four = np.zeros((ROWS, COLS), dtype=np.int8)
    board_four[2, 3:7] = 1  # SELF pieces at (2,3),(2,4),(2,5),(2,6)
    phi_s, phi_o = compute_feature_vector(board_four)
    counts = count_pattern_features(board_four)
    print(f"SELF counts: {counts['self']}")
    assert counts['self']['four'] == 1, "Expected exactly 1 four-in-a-row"
    print("✓ four-in-a-row correctly detected")

    # ---- Test 4: connected vs unconnected 2-in-a-row ----
    print("\n=== Test 4: Connected vs unconnected 2-in-a-row ===")
    # Connected: pieces at positions 0,1 within a window
    board_c = np.zeros((ROWS, COLS), dtype=np.int8)
    board_c[0, 0] = 1
    board_c[0, 1] = 1  # adjacent -> connected_2 in window [0,1,2,3]
    phi_s, phi_o = compute_feature_vector(board_c)
    c = count_pattern_features(board_c)
    print(f"Adjacent pieces: connected_2={c['self']['connected_2']}, "
          f"unconnected_2={c['self']['unconnected_2']}")
    assert c['self']['connected_2'] >= 1
    assert c['self']['unconnected_2'] == 0

    board_u = np.zeros((ROWS, COLS), dtype=np.int8)
    board_u[0, 0] = 1
    board_u[0, 2] = 1  # gap at 1 -> unconnected_2 in window [0,1,2,3]
    c2 = count_pattern_features(board_u)
    print(f"Non-adjacent pieces: connected_2={c2['self']['connected_2']}, "
          f"unconnected_2={c2['self']['unconnected_2']}")
    assert c2['self']['connected_2'] == 0
    assert c2['self']['unconnected_2'] >= 1
    print("✓ connected vs unconnected correctly distinguished")

    # ---- Test 5: opponent-blocked window doesn't count ----
    print("\n=== Test 5: Opponent-blocked window should NOT fire ===")
    board_blocked = np.zeros((ROWS, COLS), dtype=np.int8)
    board_blocked[0, 0] = 1   # SELF
    board_blocked[0, 1] = 1   # SELF
    board_blocked[0, 3] = -1  # OPP in same window [0,1,2,3]
    c3 = count_pattern_features(board_blocked)
    print(f"With OPP blocker in window: connected_2={c3['self']['connected_2']}, "
          f"unconnected_2={c3['self']['unconnected_2']}")
    # Window [0,1,2,3] has OPP at pos 3 -> NOT viable -> no feature fires
    assert c3['self']['connected_2'] == 0
    assert c3['self']['unconnected_2'] == 0
    print("✓ blocked window correctly ignored")

    # ---- Test 6: value function ----
    print("\n=== Test 6: Value function ===")
    weights = np.array([1.0, 2.0, 1.5, 5.0, 1000.0])  # [centre, c2, uc2, three, four]
    C = 2.0
    v = compute_value(board_self, weights, C)
    print(f"V(position from Test 2) = {v:.4f}")
    # Should be positive since Black has a strong three-in-a-row
    assert v > 0, "Expected positive value for Black's three-in-a-row position"
    print("✓ value function gives positive result for Black's strong position")

    print("\n=== All tests passed ===")
