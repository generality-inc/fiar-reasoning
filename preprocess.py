"""
preprocess.py
=============
Preprocessing utilities for the Four-in-a-Row model fitting pipeline.

Board representation
--------------------
We store the board as a numpy array of shape (4, 9):
    +1  = current player's piece   ("self")
    -1  = opponent's piece
     0  = empty

Using a self/opponent convention (rather than absolute Black/White)
is natural for the paper's value function, which is always written
from the perspective of the player to move:

    V(s) = sum_i w_i * [phi_i(s, self) - phi_i(s, opponent)]

FEN format (observed in the data)
----------------------------------
    "FEN: 2WBBB3/4W4/5W3/9\n\nCurrent player: Black ..."
     - Rows separated by '/'
     - 'W' = White piece, 'B' = Black piece
     - digit d = d consecutive empty squares
     - Current player follows "Current player: "
"""

import re
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROWS = 4
COLS = 9

EMPTY = 0
SELF  = 1
OPP   = -1


# ---------------------------------------------------------------------------
# FEN parsing
# ---------------------------------------------------------------------------

def parse_fen_row(row_str: str) -> list[int]:
    """
    Parse one FEN row string into a list of COLS cell values.
    Uses raw B/W labels (B=1, W=-1) before player orientation is applied.

    Returns a list of length COLS with values in {0, 1, -1}
        1  -> Black piece
       -1  -> White piece
        0  -> empty
    """
    cells = []
    for ch in row_str:
        if ch.isdigit():
            cells.extend([EMPTY] * int(ch))
        elif ch == 'B':
            cells.append(1)     # Black
        elif ch == 'W':
            cells.append(-1)    # White
        else:
            raise ValueError(f"Unexpected character in FEN row: '{ch}'")
    if len(cells) != COLS:
        raise ValueError(
            f"FEN row '{row_str}' parsed to {len(cells)} cells, expected {COLS}"
        )
    return cells


def parse_board_state(board_state_str: str):
    """
    Parse the full board_state string from the dataframe.

    Parameters
    ----------
    board_state_str : str
        e.g. "FEN: 2WBBB3/4W4/5W3/9\\n\\nCurrent player: Black ..."

    Returns
    -------
    board_abs : np.ndarray, shape (4, 9), dtype int8
        Absolute board encoding:  1=Black, -1=White, 0=empty
        Row 0 is the top row as written in FEN.

    current_player : str
        'Black' or 'White'

    board_self : np.ndarray, shape (4, 9), dtype int8
        Board re-encoded from the perspective of the current player:
            +1 = current player's piece  (SELF)
            -1 = opponent's piece        (OPP)
             0 = empty
        This is what you pass directly into the feature/value functions.
    """
    # ---- extract FEN string ----
    fen_match = re.search(r'FEN:\s*([^\n]+)', board_state_str)
    if not fen_match:
        raise ValueError(f"Could not find FEN in board_state: {board_state_str!r}")
    fen_str = fen_match.group(1).strip()

    # ---- extract current player ----
    player_match = re.search(r'Current player:\s*(Black|White)', board_state_str)
    if not player_match:
        raise ValueError(
            f"Could not find 'Current player' in board_state: {board_state_str!r}"
        )
    current_player = player_match.group(1)

    # ---- parse FEN rows ----
    row_strs = fen_str.split('/')
    if len(row_strs) != ROWS:
        raise ValueError(
            f"FEN has {len(row_strs)} rows, expected {ROWS}: '{fen_str}'"
        )

    board_abs = np.array(
        [parse_fen_row(r) for r in row_strs], dtype=np.int8
    )  # shape (4, 9)

    # ---- self/opponent orientation ----
    # In absolute encoding: Black=+1, White=-1.
    # If current player is Black, the orientation is already correct.
    # If current player is White, flip the sign so White becomes +1 (SELF).
    if current_player == 'Black':
        board_self = board_abs.copy()
    else:  # White to move
        board_self = -board_abs.copy()

    return board_abs, current_player, board_self


# ---------------------------------------------------------------------------
# Board manipulation
# ---------------------------------------------------------------------------

def place_piece(board_self: np.ndarray, row: int, col: int,
                is_self: bool = True) -> np.ndarray:
    """
    Return a new board with a piece placed at (row, col).

    Parameters
    ----------
    board_self : np.ndarray, shape (4, 9)
        Current board in self/opponent encoding.
    row, col : int
        0-indexed position (row 0 = top row in FEN).
    is_self : bool
        True  -> place current player's piece (SELF = +1)
        False -> place opponent's piece       (OPP  = -1)

    Returns
    -------
    new_board : np.ndarray, shape (4, 9)
        Updated board. The sign convention is NOT flipped — the caller is
        responsible for calling `flip_perspective` after an opponent move.
    """
    if board_self[row, col] != EMPTY:
        raise ValueError(
            f"Square ({row},{col}) is already occupied "
            f"(value={board_self[row, col]})"
        )
    new_board = board_self.copy()
    new_board[row, col] = SELF if is_self else OPP
    return new_board


def flip_perspective(board_self: np.ndarray) -> np.ndarray:
    """
    Flip the board to the opponent's perspective.
    After the current player places a piece, call this so the
    resulting board is encoded from the next player's viewpoint.
    """
    return -board_self


def apply_move(board_self: np.ndarray, row: int, col: int) -> np.ndarray:
    """
    Apply the current player's move at (row, col) and return the board
    from the *next* player's perspective (flipped).

    Equivalent to: place_piece(...) then flip_perspective(...)
    """
    new_board = place_piece(board_self, row, col, is_self=True)
    return flip_perspective(new_board)


# ---------------------------------------------------------------------------
# Move string parsing
# ---------------------------------------------------------------------------

def parse_move(move_str: str):
    """
    Parse a move string like 'm 2 5' into (row, col).

    Returns None if the move is not a standard board move
    (e.g. 'resign', None, empty string).
    """
    if not isinstance(move_str, str):
        return None
    m = re.fullmatch(r'm\s+(\d+)\s+(\d+)', move_str.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


# ---------------------------------------------------------------------------
# Tree structure
# ---------------------------------------------------------------------------

def parse_tree_node(node):
    """
    Recursively parse one node from the 'trees' or 'trees_annotated' column.

    The data uses TWO encodings for a node, which can be mixed within
    the same tree:

    Encoding A — "int-first" list  (top-level nodes and some internals)
        [int_row, int_col, child*, ...]
        e.g. [0, 5, ['2,4', [...]]]
        First two elements are Python ints; the rest are children.

    Encoding B — "string-first" list  (nodes at deeper levels)
        ['row,col', child*, ...]
        e.g. ['2,4', ['2,5', [...]]]
        First element is a string 'row,col'; the rest are children.

    Encoding C — bare string leaf
        'row,col'
        e.g. '2,4'   (a leaf with no children)

    Returns a dict:
        {
          'move': (row, col),
          'children': [...]
        }
    """
    # ---- Encoding C: bare string leaf ----
    if isinstance(node, str):
        parts = node.split(',')
        return {'move': (int(parts[0]), int(parts[1])), 'children': []}

    if not isinstance(node, list) or len(node) == 0:
        raise ValueError(f"Unexpected node: {node!r}")

    # ---- Encoding B: string-first list ['row,col', child*, ...] ----
    if isinstance(node[0], str):
        parts = node[0].split(',')
        row, col = int(parts[0]), int(parts[1])
        children = [parse_tree_node(ch) for ch in node[1:]]
        return {'move': (row, col), 'children': children}

    # ---- Encoding A: int-first list [row, col, child*, ...] ----
    row, col = int(node[0]), int(node[1])
    children = [parse_tree_node(ch) for ch in node[2:]]
    return {'move': (row, col), 'children': children}


def parse_tree(tree_list: list) -> list:
    """
    Parse the full tree (list of root nodes) for one turn.

    Parameters
    ----------
    tree_list : list
        The value of df['trees'] or df['trees_annotated'] for one row.
        Each element is a root-level move node.

    Returns
    -------
    List of parsed node dicts (one per root-level candidate move).
    """
    if not isinstance(tree_list, list) or len(tree_list) == 0:
        return []
    return [parse_tree_node(node) for node in tree_list]


# ---------------------------------------------------------------------------
# Full row preprocessing
# ---------------------------------------------------------------------------

def preprocess_row(row: dict) -> dict | None:
    """
    Preprocess one row from the dataframe.

    Parameters
    ----------
    row : dict-like
        Must have keys: 'board_state', 'extracted_move', 'trees'
        (use df.itertuples or df.to_dict('records'))

    Returns
    -------
    dict with:
        'board_abs'      : np.ndarray (4,9)  — absolute Black/White encoding
        'current_player' : str               — 'Black' or 'White'
        'board_self'     : np.ndarray (4,9)  — self/opp encoding
        'move'           : (row, col) or None
        'tree'           : list of parsed node dicts
    Returns None if board_state is missing/unparseable or move is invalid.
    """
    try:
        board_abs, current_player, board_self = parse_board_state(
            row['board_state']
        )
    except Exception:
        return None

    move = parse_move(row.get('extracted_move'))
    tree = parse_tree(row.get('trees', []))

    return {
        'board_abs':       board_abs,
        'current_player':  current_player,
        'board_self':      board_self,
        'move':            move,
        'tree':            tree,
    }


# ---------------------------------------------------------------------------
# Pretty-print utility
# ---------------------------------------------------------------------------

def board_to_str(board_self: np.ndarray, current_player: str = '?') -> str:
    """
    Human-readable board display using self/opp encoding.

    S = current player's piece
    O = opponent's piece
    . = empty
    """
    symbol = {SELF: 'S', OPP: 'O', EMPTY: '.'}
    rows_str = []
    for r in range(ROWS):
        row_str = ' '.join(symbol[v] for v in board_self[r])
        rows_str.append(f"  {r}  {row_str}")
    col_header = '     ' + ' '.join(str(c) for c in range(COLS))
    return (
        f"Current player: {current_player}\n"
        + col_header + '\n'
        + '\n'.join(rows_str)
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Simulate a board_state string we observed in the notebook output
    test_board_state = (
        "FEN: 2WBBB3/4W4/5W3/9\n\n"
        "Current player: Black (you are Black)"
    )

    board_abs, player, board_self = parse_board_state(test_board_state)
    print("=== parse_board_state ===")
    print(f"current_player : {player}")
    print(f"board_abs:\n{board_abs}")
    print(f"\nboard_self (Black to move, so same as abs):\n{board_self}")
    print()
    print(board_to_str(board_self, player))

    # After Black plays at (0, 0), flip to White's perspective
    print("\n=== after Black plays (0,0), White's perspective ===")
    next_board = apply_move(board_self, row=0, col=0)
    print(next_board)

    # Test move parsing
    print("\n=== parse_move ===")
    for s in ['m 2 5', 'm 0 0', 'resign', None, 'm2 3']:
        print(f"  {s!r:12s}  ->  {parse_move(s)}")

    # Test tree parsing
    print("\n=== parse_tree ===")
    raw_tree = [[0, 5, [0, 6]], [1, 3], [1, 5]]
    parsed = parse_tree(raw_tree)
    for node in parsed:
        print(f"  root move {node['move']}, children: {[c['move'] for c in node['children']]}")

    # Test with the deeper tree from the notebook output
    raw_tree2 = [[3, 2, [0, 5, ['2,4', ['2,5', ['3,4', ['0,4']]]]]]]
    parsed2 = parse_tree(raw_tree2)
    def show_tree(node, indent=0):
        print(' ' * indent + str(node['move']))
        for ch in node['children']:
            show_tree(ch, indent + 2)
    print("\n  Deeper tree:")
    for n in parsed2:
        show_tree(n, indent=2)
