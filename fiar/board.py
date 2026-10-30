"""
Board representation, FEN parsing and search-tree parsing.

Boards are (4, 9) int8 arrays encoded from the perspective of the player to move:
    +1 = own piece (SELF), -1 = opponent piece (OPP), 0 = empty.
Row 0 is the top row as written in FEN.
"""

import re

import numpy as np

ROWS, COLS = 4, 9
EMPTY, SELF, OPP = 0, 1, -1

_MOVE_RE = re.compile(r'm\s+(\d+)\s+(\d+)')


def parse_board_state(board_state: str):
    """
    Parse a board_state string such as
    "FEN: 2WBBB3/4W4/5W3/9\\n\\nCurrent player: Black (B)\\n".

    Returns (board_abs, current_player, board_self), where board_abs uses
    Black = +1 / White = -1 and board_self is from the current player's view.
    """
    fen = re.search(r'FEN:\s*([^\n]+)', board_state)
    player = re.search(r'Current player:\s*(Black|White)', board_state)
    if not fen or not player:
        raise ValueError(f'Cannot parse board_state: {board_state!r}')

    rows = fen.group(1).strip().split('/')
    if len(rows) != ROWS:
        raise ValueError(f'FEN has {len(rows)} rows, expected {ROWS}')

    board_abs = np.zeros((ROWS, COLS), dtype=np.int8)
    for r, row in enumerate(rows):
        cells = []
        for ch in row:
            if ch.isdigit():
                cells.extend([EMPTY] * int(ch))
            elif ch in 'BW':
                cells.append(1 if ch == 'B' else -1)
            else:
                raise ValueError(f'Unexpected character in FEN row: {ch!r}')
        if len(cells) != COLS:
            raise ValueError(f'FEN row {row!r} has {len(cells)} cells, expected {COLS}')
        board_abs[r] = cells

    current_player = player.group(1)
    board_self = board_abs.copy() if current_player == 'Black' else -board_abs
    return board_abs, current_player, board_self


def apply_move(board_self: np.ndarray, row: int, col: int) -> np.ndarray:
    """Place the mover's piece at (row, col); return the board from the next player's view."""
    if board_self[row, col] != EMPTY:
        raise ValueError(f'Square ({row},{col}) is occupied')
    board = board_self.copy()
    board[row, col] = SELF
    return -board


def parse_move(move: str):
    """'m 2 5' -> (2, 5); anything else -> None."""
    if not isinstance(move, str):
        return None
    m = _MOVE_RE.fullmatch(move.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def _parse_node(node) -> dict:
    """
    Parse one extracted-tree node into {'move': (r, c), 'children': [...]}.

    Nodes are lists ['r,c', child, ...]; older rows use [r, c, child, ...],
    and a bare 'r,c' string is a leaf.
    """
    if isinstance(node, str):
        r, c = node.split(',')
        return {'move': (int(r), int(c)), 'children': []}
    if not isinstance(node, list) or not node:
        raise ValueError(f'Unexpected node: {node!r}')
    if isinstance(node[0], str):
        r, c = node[0].split(',')
        children = node[1:]
    else:
        r, c = node[0], node[1]
        children = node[2:]
    return {'move': (int(r), int(c)), 'children': [_parse_node(ch) for ch in children]}


def parse_tree(trees) -> list:
    """Parse an extracted search tree (list of depth-1 nodes)."""
    if not isinstance(trees, list):
        return []
    return [_parse_node(node) for node in trees]
